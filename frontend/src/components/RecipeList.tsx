import type { Recipe } from '../api'

interface Props {
  recipes: Recipe[]
}

export function RecipeList({ recipes }: Props) {
  return (
    <section>
      <h2>Recipes</h2>
      {recipes.length === 0 ? (
        <p className="muted">No recipes yet.</p>
      ) : (
        <ul className="cards">
          {recipes.map((recipe) => (
            <li key={recipe.id}>
              <h3>{recipe.title}</h3>
              {recipe.ingredients.length > 0 && (
                <p className="muted">
                  {recipe.ingredients.map((item) => item.ingredient.name).join(', ')}
                </p>
              )}
              <p>{recipe.instructions}</p>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
