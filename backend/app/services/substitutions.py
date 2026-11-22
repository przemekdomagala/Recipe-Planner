"""Curated ingredient substitutions used to adapt recipes to a dietary profile.

The table maps an ingredient phrase to replacement candidates, best first. A lookup picks
the longest phrase contained in the ingredient name ("unsalted butter" uses "butter") and
keeps only the candidates that pass the profile, so a replacement can never reintroduce a
forbidden ingredient. The table also serves as retrieved knowledge in the LLM prompt.
"""

from collections.abc import Collection

from app.services.dietary import DietaryProfile, check_ingredient
from app.services.taxonomy import contains_phrase, same_ingredient, tokenize


def _table(*groups: tuple[str, tuple[str, ...]]) -> dict[tuple[str, ...], tuple[str, ...]]:
    """Build the lookup table from `("phrase | phrase", replacements)` groups."""
    table: dict[tuple[str, ...], tuple[str, ...]] = {}
    for phrases, replacements in groups:
        for phrase in phrases.split("|"):
            table[tokenize(phrase.strip())] = replacements
    return table


_VEG_PROTEIN = ("tofu", "chickpeas", "mushrooms")
_GF_PASTA = ("gluten free pasta", "zucchini noodles")

SUBSTITUTES = _table(
    # Dairy
    ("butter | salted butter | chilled butter | melted butter", ("olive oil", "vegan butter")),
    ("ghee", ("coconut oil", "vegetable oil")),
    ("milk | whole milk | semi skimmed milk", ("oat milk", "soya milk", "coconut milk")),
    ("buttermilk | sour milk", ("soya milk with lemon juice", "oat milk with lemon juice")),
    (
        "cream | double cream | heavy cream | single cream | whipping cream | clotted cream",
        ("coconut cream", "cashew cream"),
    ),
    ("sour cream | creme fraiche | fromage frais", ("coconut yogurt", "cashew cream")),
    ("yogurt | yoghurt | greek yogurt", ("coconut yogurt", "soya yogurt")),
    ("cheese | cheddar | gouda | gruyere | monterey jack", ("vegan cheese", "nutritional yeast")),
    ("parmesan | parmigiano | pecorino", ("nutritional yeast", "vegan parmesan")),
    ("mozzarella", ("vegan mozzarella",)),
    ("feta", ("vegan feta",)),
    ("paneer", ("firm tofu", "chickpeas")),
    (
        "ricotta | cottage cheese | cream cheese | mascarpone | quark",
        ("silken tofu", "vegan cream cheese"),
    ),
    ("milk chocolate | white chocolate", ("dark chocolate",)),
    ("condensed milk", ("coconut condensed milk",)),
    # Eggs and honey
    ("egg | eggs", ("flax egg", "chia egg")),
    ("egg white | egg yolk", ("aquafaba",)),
    ("mayonnaise", ("vegan mayonnaise",)),
    ("honey", ("maple syrup", "agave syrup", "sugar free syrup")),
    # Meat and fish
    ("chicken | turkey | duck", (*_VEG_PROTEIN, "jackfruit")),
    (
        "beef | minced beef | ground beef | steak | veal | lamb | lamb mince | mince | pork"
        " | ground pork | minced pork | goat meat | turkey mince",
        ("lentils", "mushrooms", "tempeh"),
    ),
    (
        "bacon | ham | prosciutto | pancetta | chorizo | sausage | lardon",
        ("smoked tofu", "mushrooms"),
    ),
    (
        "chicken stock | beef stock | lamb stock | fish stock | seafood stock | shrimp stock"
        " | chicken bouillon | beef gravy",
        ("vegetable stock",),
    ),
    (
        "fish | white fish | cod | salmon | haddock | hake | tuna | mackerel | trout"
        " | sea bass | snapper",
        ("tofu", "chickpeas", "chicken breast"),
    ),
    ("anchovy", ("capers", "olives")),
    ("fish sauce | worcestershire sauce", ("soy sauce", "tamari", "coconut aminos")),
    ("oyster sauce", ("hoisin sauce", "soy sauce", "coconut aminos")),
    (
        "prawn | shrimp | crab | lobster | squid | mussel | clam | scallop | seafood",
        ("king oyster mushrooms", "tofu", "chicken breast"),
    ),
    ("shrimp paste | dried shrimp | prahok", ("miso paste", "salt")),
    ("curry paste", ("vegan curry paste",)),
    ("gelatine | gelatin", ("agar agar",)),
    ("lard | suet | duck fat | goose fat | lamb fat", ("vegetable shortening", "vegetable oil")),
    # Gluten
    (
        "flour | plain flour | all purpose flour | self raising flour | bread flour"
        " | wheat flour | strong white flour",
        ("gluten free flour", "rice flour", "almond flour"),
    ),
    (
        "pasta | spaghetti | penne | macaroni | fettuccine | linguine | tagliatelle | farfalle"
        " | rigatoni | pappardelle | paccheri | lasagne",
        _GF_PASTA,
    ),
    ("noodle | egg noodle | udon", ("rice noodles", "zucchini noodles")),
    (
        "bread | baguette | ciabatta | bun | bread roll | toast | naan | pita",
        ("gluten free bread", "lettuce wraps"),
    ),
    ("tortilla | flour tortilla", ("corn tortillas", "lettuce wraps")),
    ("breadcrumb | panko", ("gluten free breadcrumbs", "ground almonds", "crushed pork rinds")),
    ("couscous | bulgur | freekeh | semolina", ("quinoa", "cauliflower rice")),
    ("soy sauce | dark soy sauce", ("tamari", "coconut aminos")),
    ("beer | stout", ("gluten free beer", "vegetable stock")),
    ("oat | rolled oat | porridge oat | oatmeal", ("gluten free oats", "quinoa flakes")),
    ("pastry | puff pastry | filo pastry | shortcrust pastry", ("gluten free pastry",)),
    ("biscuit | digestive biscuit | cracker", ("gluten free biscuits", "ground almonds")),
    # High-carb (keto)
    ("rice | basmati rice | jasmine rice | long grain rice", ("cauliflower rice",)),
    ("rice noodle | vermicelli", ("zucchini noodles", "shirataki noodles")),
    ("potato | new potato | sweet potato", ("cauliflower", "turnips", "celeriac")),
    ("sugar | caster sugar | brown sugar | icing sugar", ("erythritol", "stevia")),
    ("maple syrup | golden syrup | syrup", ("sugar free syrup",)),
    ("corn tortilla", ("lettuce wraps",)),
    ("cornstarch | corn flour | cornflour | potato starch", ("xanthan gum",)),
    (
        "chickpea | lentil | kidney bean | black bean | pinto bean | cannellini bean"
        " | butter bean | haricot bean | baked bean",
        ("mushrooms", "green beans"),
    ),
    # Nuts, seeds, soy, mustard, celery, sulphites
    ("peanut | roasted peanut", ("sunflower seeds", "pumpkin seeds")),
    ("peanut butter", ("sunflower seed butter", "tahini")),
    ("peanut oil | ground nut oil", ("sunflower oil", "vegetable oil")),
    (
        "almond | cashew | walnut | pecan | hazelnut | pistachio | pine nut | nut",
        ("sunflower seeds", "pumpkin seeds"),
    ),
    ("almond milk", ("oat milk", "soya milk")),
    ("almond extract | almond essence", ("vanilla extract",)),
    ("tofu | firm tofu | silken tofu", ("chickpeas", "paneer")),
    ("tempeh", ("mushrooms", "chickpeas")),
    ("soya milk | soy milk", ("oat milk", "rice milk")),
    ("miso", ("tahini", "salt")),
    ("tamari", ("coconut aminos",)),
    ("hoisin sauce", ("plum sauce",)),
    ("sesame oil | sesame seed oil", ("sunflower oil", "olive oil")),
    ("sesame | sesame seed", ("sunflower seeds", "poppy seeds")),
    ("tahini", ("sunflower seed butter", "greek yogurt")),
    ("mustard | dijon mustard | english mustard | wholegrain mustard", ("horseradish", "wasabi")),
    ("mustard seed", ("cumin seeds",)),
    ("celery", ("fennel", "bok choi")),
    ("celery salt", ("salt",)),
    ("celeriac", ("parsnips",)),
    (
        "wine | red wine | white wine | dry white wine | cooking wine | sherry | dry sherry",
        ("vegetable stock",),
    ),
    ("wine vinegar | balsamic vinegar | sherry vinegar", ("lemon juice", "apple cider vinegar")),
    ("dried apricot", ("fresh apricots",)),
)


def candidates_for(name: str) -> tuple[str, ...]:
    """Return the table's replacement candidates for the longest phrase found in `name`."""
    tokens = tokenize(name)
    best: tuple[str, ...] = ()
    best_size = 0
    for phrase, replacements in SUBSTITUTES.items():
        if len(phrase) > best_size and contains_phrase(tokens, phrase):
            best, best_size = replacements, len(phrase)
    return best


def allowed_substitutes(
    name: str, profile: DietaryProfile, *, prefer: Collection[str] = ()
) -> list[str]:
    """Return the replacements for `name` that satisfy the profile, best first.

    Args:
        name: The ingredient to replace.
        profile: The constraints every replacement must pass.
        prefer: Ingredients the user already has (e.g. the fridge); matching replacements
            are moved to the front.

    Returns:
        Allowed replacements; empty if the table has none for this ingredient.
    """
    allowed = [r for r in candidates_for(name) if check_ingredient(r, profile) is None]
    in_stock = [r for r in allowed if any(same_ingredient(r, item) for item in prefer)]
    return in_stock + [r for r in allowed if r not in in_stock]
