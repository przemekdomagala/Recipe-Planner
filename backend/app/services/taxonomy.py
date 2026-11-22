"""Ingredient taxonomy: deterministic, keyword-based tagging of ingredient names.

Every ingredient name is tokenized (lowercase, accents stripped, simple singular forms)
and matched against curated phrases, longest phrase first. Tokens claimed by a longer
phrase are not matched again, which is how "peanut butter" avoids the dairy tag of
"butter" and "coconut milk" avoids the one of "milk". Qualifiers such as "gluten free" or
"vegan" remove tags from the rest of the name.

The rules were curated against TheMealDB's ingredient list. Names no rule recognises are
reported as *unclassified* (see `python -m app.cli unclassified`) rather than assumed safe.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache


class Tag(StrEnum):
    """A property of an ingredient that diets or allergies care about."""

    # The 14 allergens of EU Regulation 1169/2011, Annex II.
    GLUTEN = "gluten"
    CRUSTACEANS = "crustaceans"
    EGGS = "eggs"
    FISH = "fish"
    PEANUTS = "peanuts"
    SOY = "soy"
    MILK = "milk"
    TREE_NUTS = "tree_nuts"
    CELERY = "celery"
    MUSTARD = "mustard"
    SESAME = "sesame"
    SULPHITES = "sulphites"
    LUPIN = "lupin"
    MOLLUSCS = "molluscs"
    # Diet-relevant properties.
    MEAT = "meat"
    ANIMAL_BYPRODUCT = "animal_byproduct"  # gelatine, lard, suet: from slaughtered animals
    HONEY = "honey"
    HIGH_CARB = "high_carb"


_T = Tag
_NONE: frozenset[Tag] = frozenset()
_ANIMAL = frozenset(
    {_T.MEAT, _T.FISH, _T.CRUSTACEANS, _T.MOLLUSCS, _T.ANIMAL_BYPRODUCT, _T.MILK, _T.EGGS}
)


# Words that only describe form, cut or variety; ignored when matching fridge items to
# recipe ingredients ("chicken" covers "chicken thighs", "onion" covers "red onions").
def _words(text: str) -> frozenset[str]:
    """Return the whitespace-separated words of `text` as a set."""
    return frozenset(text.split())


DESCRIPTORS = _words(
    """
    baby beaten boneless brown can chopped cooked crushed cubed diced dried extra fillet
    fine finely free fresh freshly frozen grated green ground half large lean leg medium
    minced new organic peeled range raw red ripe roasted salted semi shredded skimmed
    skinless sliced small smoked tinned unsalted unwaxed virgin white whole yellow yolk
    breast thigh drumstick wing of king tiger jumbo
    """
)

# Staples assumed to be in every kitchen: not counted as missing or towards coverage.
_STAPLE_CORE = _words("salt pepper water oil ice")
_STAPLE_WORDS = _STAPLE_CORE | _words(
    """
    black white ground cold warm boiling hot sea kosher coarse table olive vegetable
    sunflower canola rapeseed extra virgin fine freshly cracked cube cooking high heat
    """
)

_IRREGULAR = {
    "leaves": "leaf",
    "loaves": "loaf",
    "halves": "half",
    "chillies": "chilli",
    "cookies": "cookie",
    "brownies": "brownie",
}


def _singular(word: str) -> str:
    """Return a crude singular form, applied identically to names and rule phrases."""
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if len(word) > 4 and word.endswith(("ches", "shes", "xes")):
        return word[:-2]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("oes"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


@lru_cache(maxsize=8192)
def tokenize(name: str) -> tuple[str, ...]:
    """Split an ingredient name into normalized tokens.

    Args:
        name: Free-text ingredient name, e.g. `"Free-range Eggs, beaten"`.

    Returns:
        Lowercase, accent-free, singular tokens, e.g. `("free", "range", "egg", "beaten")`.
    """
    text = unicodedata.normalize("NFKD", name.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return tuple(_singular(word) for word in re.split(r"[^a-z0-9]+", text) if word)


# Synonyms and forms that denote the same ingredient when matching fridge items.
_ALIASES: dict[tuple[str, ...], tuple[str, ...]] = {
    tokenize(alias): tokenize(canonical)
    for aliases, canonical in (
        ("garlic clove | garlic bulb | garlic granule", "garlic"),
        ("scallion", "spring onion"),
        ("cilantro | cilantro leaf | coriander leaf", "coriander"),
        ("courgette", "zucchini"),
        ("aubergine | egg plant", "eggplant"),
        ("chili", "chilli"),
        ("yoghurt", "yogurt"),
        ("garbanzo bean", "chickpea"),
        ("beef mince", "beef"),
        ("lamb mince", "lamb"),
        ("pork mince", "pork"),
        ("turkey mince", "turkey"),
        ("basil leaf", "basil"),
        ("parsley leaf", "parsley"),
    )
    for alias in aliases.split("|")
}


def contains_phrase(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    """Return whether `phrase` occurs as a contiguous run in `tokens`."""
    size = len(phrase)
    return any(tokens[i : i + size] == phrase for i in range(len(tokens) - size + 1))


def core_tokens(name: str) -> tuple[str, ...]:
    """Return the name's tokens without descriptors (all tokens if nothing else is left).

    Known synonyms are mapped to one spelling, e.g. "garlic clove" and "cilantro".
    """
    tokens = tokenize(name)
    core = tuple(token for token in tokens if token not in DESCRIPTORS) or tokens
    return _ALIASES.get(core, core)


def same_ingredient(a: str, b: str) -> bool:
    """Return whether two names denote the same ingredient, ignoring descriptors.

    Deliberately conservative: "chicken" matches "chicken breasts" but not "chicken stock",
    and "butter" does not match "peanut butter".
    """
    return core_tokens(a) == core_tokens(b)


def is_staple(name: str) -> bool:
    """Return whether the ingredient is a kitchen staple such as salt, pepper, water or oil."""
    tokens = set(tokenize(name))
    return bool(tokens & _STAPLE_CORE) and tokens <= _STAPLE_WORDS


def _rules(*groups: tuple[Iterable[Tag], str]) -> dict[tuple[str, ...], frozenset[Tag]]:
    """Build the phrase table from `(tags, "phrase | phrase | ...")` groups."""
    table: dict[tuple[str, ...], frozenset[Tag]] = {}
    for tags, phrases in groups:
        for phrase in phrases.split("|"):
            key = tokenize(phrase.strip())
            if key:
                table[key] = table.get(key, _NONE) | frozenset(tags)
    return table


_RULES = _rules(
    # --- Meat, fish and other animal products ------------------------------------------
    (
        {_T.MEAT},
        "beef | steak | veal | pork | pig | ham | bacon | lardon | pancetta | prosciutto"
        " | chorizo | salami | pepperoni | sausage | kielbasa | kabanos | morcilla"
        " | black pudding | lamb | mutton | goat meat | chicken | turkey | duck | goose"
        " | mince | meat | oxtail | tripe | liver | kidney | brisket"
        " | sirloin | chuck roast | rib | trotter | shank | doner | venison | rabbit | frog"
        " | gravy | pork rind | bone marrow | jamon",
    ),
    ({_T.ANIMAL_BYPRODUCT}, "gelatine | gelatin | lard | suet | duck fat | goose fat | lamb fat"),
    ({_T.ANIMAL_BYPRODUCT, _T.HIGH_CARB}, "mincemeat"),
    (
        {_T.FISH},
        "fish | salmon | cod | codfish | haddock | hake | tuna | mackerel | herring | sardine"
        " | pilchard | anchovy | trout | sea bass | snapper | barramundi | monkfish"
        " | worcestershire sauce | prahok | fish sauce",
    ),
    (
        {_T.CRUSTACEANS},
        "prawn | shrimp | crab | lobster | crayfish | langoustine | shrimp paste",
    ),
    ({_T.MOLLUSCS}, "squid | calamari | clam | mussel | oyster | scallop | octopus | conch"),
    ({_T.MOLLUSCS, _T.GLUTEN}, "oyster sauce"),
    ({_T.FISH, _T.CRUSTACEANS, _T.MOLLUSCS}, "seafood"),
    # Thai curry pastes are traditionally made with shrimp paste.
    ({_T.CRUSTACEANS}, "curry paste"),
    ({_T.HONEY, _T.HIGH_CARB}, "honey"),
    # --- Eggs and milk -------------------------------------------------------------------
    ({_T.EGGS}, "egg | egg white | egg yolk | egg wash | mayonnaise | aioli"),
    ({_T.EGGS, _T.GLUTEN, _T.HIGH_CARB}, "egg noodle | egg roll | egg roll wrapper"),
    ({_T.EGGS, _T.HIGH_CARB}, "meringue"),
    ({_T.EGGS, _T.MILK, _T.HIGH_CARB}, "custard"),
    (
        {_T.MILK},
        "milk | butter | cream | cheese | yogurt | yoghurt | ghee | buttermilk | creme fraiche"
        " | fromage frais | mascarpone | ricotta | mozzarella | mozarella | parmesan"
        " | parmigiano | pecorino | cheddar | feta | brie | gouda | gruyere | emmentaler"
        " | manchego | stilton | paneer | quark | queso | curd | malai | whey | bryndza"
        " | vasterbottensost | sour cream | clotted cream | double cream | single cream"
        " | heavy cream | whipping cream | cream cheese | evaporated milk",
    ),
    (
        {_T.MILK, _T.HIGH_CARB},
        "condensed milk | dulce de leche | ice cream | milk chocolate | white chocolate"
        " | caramel sauce | caramelized sugar sauce | toffee | mars bar",
    ),
    ({_T.MILK, _T.TREE_NUTS, _T.SOY, _T.HIGH_CARB}, "nutella"),
    ({_T.MILK, _T.TREE_NUTS}, "pesto"),
    # --- Gluten and starch ----------------------------------------------------------------
    (
        {_T.GLUTEN, _T.HIGH_CARB},
        "flour | wheat | whole wheat | bulgur | freekeh | semolina | couscous | khus khus | rye"
        " | barley | spelt | pasta | spaghetti | macaroni | penne | rigatoni | farfalle"
        " | fettuccine | linguine | tagliatelle | pappardelle | paccheri | lasagne | lasagna"
        " | vermicelli | noodle | udon | fideo | sevaiiya | bread | baguette | ciabatta | bun"
        " | bread roll | toast | naan | pita | tortilla | muffin | pastry | phyllo | filo"
        " | wonton | breadcrumb | panko | biscuit | cookie | cracker | pretzel | oat | oatmeal"
        " | cereal | rice krispy | mixed grain | christmas pudding | seitan | gnocchi"
        " | digestive biscuit | grape nut",
    ),
    ({_T.GLUTEN, _T.SULPHITES, _T.HIGH_CARB}, "beer | stout"),
    ({_T.GLUTEN}, "malt vinegar"),
    (
        {_T.HIGH_CARB},
        "rice | quinoa | potato | sweet potato | yam | cassava | yuca | taro | tapioca"
        " | corn | sweetcorn | cornmeal | polenta | grits | masarepa | arepa | casabe"
        " | breadfruit | plantain | banana | sugar | syrup | molasses | treacle | jam"
        " | marmalade | date | raisin | sultana | prune | dried fruit | candied peel"
        " | mixed peel | glace cherry | dried cranberry | dried cherry | chocolate"
        " | chocolate chip | caramel | marshmallow | turkish delight | fries | ketchup"
        " | barbeque sauce | sweet chilli sauce | plum sauce | duck sauce | juice | mirin"
        " | custard powder | corn tortilla | taco shell | rice paper | rice noodle"
        " | rice stick noodle | rice vermicelli | vermicelli rice noodle | glass noodle"
        " | chickpea | lentil | dal | kidney bean | black bean | pinto bean | cannellini bean"
        " | borlotti bean | haricot bean | navy bean | butter bean | lima bean"
        " | great northern bean | refried bean | baked bean | canned bean | black eyed pea"
        " | split pea | mung bean | dried white bean | cornstarch | corn flour | cornflour"
        " | rice flour | potato starch | buckwheat | buckwheat flour | cassava flour"
        " | chickpea flour | farine | coco sugar | palm sugar | papelon | cranberry sauce"
        " | starch | falafel",
    ),
    ({_T.GLUTEN, _T.HIGH_CARB, _T.TREE_NUTS}, "nougatine"),
    # --- Nuts, seeds, soy, mustard, celery, lupin -------------------------------------------
    ({_T.PEANUTS}, "peanut | groundnut | ground nut | peanut oil | peanut butter | satay"),
    ({_T.PEANUTS, _T.HIGH_CARB}, "peanut brittle | peanut cookie"),
    (
        {_T.TREE_NUTS},
        "almond | cashew | walnut | pecan | hazelnut | hazlenut | pistachio | macadamia"
        " | brazil nut | pine nut | praline | nut | frangipane",
    ),
    ({_T.TREE_NUTS, _T.HIGH_CARB}, "marzipan | almond paste"),
    (
        {_T.SOY},
        "soy | soya | soya bean | tofu | tempeh | edamame | miso | tamari | fermented black bean",
    ),
    ({_T.SOY, _T.GLUTEN}, "soy sauce | doubanjiang | chilli bean paste"),
    ({_T.SOY, _T.GLUTEN, _T.HIGH_CARB}, "hoisin sauce | gochujang"),
    ({_T.SESAME}, "sesame | sesame seed | sesame oil | tahini"),
    ({_T.SESAME, _T.HIGH_CARB}, "hummus"),
    ({_T.MUSTARD}, "mustard | mustard seed | mustard powder | panch poran"),
    ({_T.CELERY}, "celery | celeriac | celery salt | celery seed"),
    ({_T.LUPIN}, "lupin | lupini"),
    (
        {_T.SULPHITES},
        "wine | sherry | port | vermouth | wine vinegar | balsamic vinegar | cider"
        " | red wine jelly",
    ),
    ({_T.SULPHITES, _T.HIGH_CARB}, "dried apricot"),
    # --- Overrides: names that contain a tagged word but don't have its property -----------
    (
        (),
        "coconut milk | coconut cream | coconut yogurt | coconut oil | cocoa butter"
        " | palm butter | seed butter | sunflower seed butter | pumpkin seed butter"
        " | cream of tartar | egg plant | eggplant | flax egg"
        " | chia egg | oyster mushroom | fish seasoning | vegetable milk | vegetable shortening"
        " | cauliflower rice | zucchini noodle | courgette noodle | shirataki noodle | butternut"
        " | water chestnut | chestnut | nutmeg | coconut | rice vinegar | cider vinegar"
        " | apple cider vinegar | coconut aminos | lemon juice | lime juice | pickle juice"
        " | beef tomato | gingerbread spice | coconut flake | stir fry | banana pepper"
        " | madras paste",
    ),
    ({_T.TREE_NUTS}, "almond milk | almond butter | almond flour | cashew cream"),
    ({_T.SOY}, "soya milk | soy milk | soya yogurt | soy yogurt"),
    ({_T.HIGH_CARB}, "rice milk | rice wine | coconut condensed milk"),
    ({_T.GLUTEN, _T.HIGH_CARB}, "oat milk"),
    ({_T.GLUTEN, _T.SULPHITES}, "shaoxing wine"),
    # --- Known ingredients without any tracked property ------------------------------------
    (
        (),
        "salt | pepper | water | oil | olive oil | vegetable oil | sunflower oil | rapeseed oil"
        " | canola oil | truffle oil | ice | onion | shallot | challot | garlic | leek | scallion"
        " | spring onion | chive | tomato | passata | tomato puree | carrot | parsnip | turnip"
        " | swede | vegetable | brussels sprout | yucca | beetroot | beet | radish | daikon"
        " | cabbage | lettuce | spinach | kale"
        " | rocket | broccoli | cauliflower | courgette | zucchini | aubergine | cucumber"
        " | mushroom | asparagus | avocado | squash | pumpkin | okra | artichoke | fennel"
        " | green bean | runner bean | broad bean | bean sprout | pea | petit pois | snow pea"
        " | long bean | chinese leaf | berry | mixed berry | avacado | cardomom | seed"
        " | shortening | chimichurri | pico de gallo | hotsauce | tobasco | sake | ackee"
        " | sugar snap pea | pak choi | bok choi | pak koi | callaloo | sauerkraut | olive"
        " | caper | gherkin | pickle | lemon | lime | orange | apple | pear | peach | apricot"
        " | plum | cherry | grape | grapefruit | strawberry | raspberry | blueberry | blackberry"
        " | redcurrant | currant | cranberry | pomegranate | rhubarb | mango | pineapple | fig"
        " | passion fruit | kiwi | melon | watermelon | papaya | longan | basil | parsley"
        " | coriander | cilantro | dill | mint | thyme | rosemary | sage | oregano | marjoram"
        " | tarragon | bay leaf | cumin | paprika | turmeric | cinnamon | clove | cardamom"
        " | ginger | chilli | chili | jalapeno | habanero | scotch bonnet | cayenne | allspice"
        " | saffron | star anise | fenugreek | garam masala | curry powder | sumac | vanilla"
        " | five spice | juniper | peppercorn | lemongrass | galangal | kaffir lime | makrut"
        " | nutritional yeast | yeast | baking powder | bicarbonate of soda | baking soda"
        " | cocoa | cacao | coffee | tea | vinegar | white vinegar | stock | vegetable stock"
        " | aquafaba | agar agar | erythritol | stevia | xanthan gum | salsa | sriracha"
        " | tabasco | harissa | hot sauce | chilli sauce | tomato sauce | food colouring"
        " | horseradish | wasabi | sunflower seed | pumpkin seed | poppy seed | chia seed"
        " | flaxseed | tamarind | pandan | banana leaf | vine leaf | grape leaf | rose water"
        " | orange blossom water | seasoning | italian seasoning | cajun | jerk | sazon"
        " | ras el hanout | za atar | bouquet garni | mixed spice | caraway | nigella"
        " | sichuan pepper | szechuan peppercorn | kampot pepper | pul biber | annatto"
        " | achiote | mahleb | savory | summer savoury | sorrel | jackfruit | palm heart"
        " | bamboo shoot | lemon zest | orange zest | soda water | rum | brandy | lettuce wrap",
    ),
)

# Qualifiers remove tags from the rest of the name ("gluten free pasta", "vegan butter").
_QUALIFIERS: dict[tuple[str, ...], frozenset[Tag]] = {
    tokenize("gluten free"): frozenset({_T.GLUTEN}),
    tokenize("dairy free"): frozenset({_T.MILK}),
    tokenize("egg free"): frozenset({_T.EGGS}),
    tokenize("sugar free"): frozenset({_T.HIGH_CARB}),
    tokenize("vegan"): _ANIMAL | {_T.HONEY},
    tokenize("plant based"): _ANIMAL | {_T.HONEY},
    tokenize("vegetarian"): frozenset({_T.MEAT, _T.FISH, _T.CRUSTACEANS, _T.MOLLUSCS}),
}
_MAX_PHRASE = max(len(key) for key in (*_RULES, *_QUALIFIERS))


@dataclass(frozen=True, slots=True)
class Classification:
    """The tags found in an ingredient name and whether any rule recognised it."""

    tags: frozenset[Tag]
    known: bool


@lru_cache(maxsize=8192)
def classify(name: str) -> Classification:
    """Tag an ingredient name using the curated phrase rules.

    Args:
        name: Free-text ingredient name.

    Returns:
        The matched tags; `known` is `False` when no rule matched any part of the name, so
        the caller can't tell a safe ingredient from an unrecognised one.
    """
    tokens = tokenize(name)
    claimed = [False] * len(tokens)
    tags: set[Tag] = set()
    removed: set[Tag] = set()
    known = False
    for size in range(min(_MAX_PHRASE, len(tokens)), 0, -1):
        for start in range(len(tokens) - size + 1):
            span = range(start, start + size)
            if any(claimed[i] for i in span):
                continue
            key = tokens[start : start + size]
            if key in _QUALIFIERS:
                removed |= _QUALIFIERS[key]
            elif key in _RULES:
                tags |= _RULES[key]
                known = True
            else:
                continue
            for i in span:
                claimed[i] = True
    return Classification(tags=frozenset(tags - removed), known=known)
