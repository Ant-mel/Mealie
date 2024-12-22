from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import ChatOllama
import json
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Meal Planner API")

# Add CORS Middleware
origins = ["*"]  # Replace with specific origins if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MealRequest(BaseModel):
    prompt: str

# Initialize LLM
llm = ChatOllama(
    model="llama3",
    temperature=0
)

@app.post("/generate_meals")
async def generate_meals(request: MealRequest):
    try:
        messages = [
            (
                "system",
                """You are a helpful assistant that generates meal plans on request.
                When giving Meals and their lists of ingredients, you always answer only in a json format.

                Never input extra text.

                The response should include:
                - Recipe name
                - List of ingredients with specific measurements (e.g., cups, grams, tablespoons)
                - Preparation time in minutes
                - Number of servings

                Example:
                input: Give me 2 breakfast ideas
                output:
                {"recipes": [{
                "name": "Classic Chicken Salad",
                "ingredients": [
                    "2 cups cooked chicken, diced",
                    "2 cups fresh spinach",
                    "1 cup cherry tomatoes, halved",
                    "1 medium bell pepper, sliced",
                    "2 tablespoons olive oil",
                    "1 teaspoon salt"
                ],
                "prep_time": 20,
                "servings": 2
                },
                {
                "name": "Spicy Kimchi Jjigae",
                "ingredients": [
                    "2 cups ripe kimchi",
                    "200g pork belly, sliced",
                    "400g firm tofu, cubed",
                    "3 cloves garlic, minced",
                    "2 tablespoons gochugaru",
                    "4 cups water",
                    "1 tablespoon soy sauce"
                ],
                "prep_time": 45,
                "servings": 4
                }]}

                Avoid this:
                input: Give me 2 breakfast ideas
                output:
                Here are two healthy breakfasts for you

                {"recipes": [{...}]}
                """
            ),
            ("human", request.prompt)
        ]

        print(request.prompt)

        # Get response from LLM
        ai_msg = llm.invoke(messages)
        print('llm job done')

        print(ai_msg.content)

        json_data = json.loads(ai_msg.content)

        # # Parse the response to ensure it's valid JSON
        data = pd.DataFrame(json_data['recipes'])
        # response_data = json.loads(str(ai_msg))
        print(data['name'][0])

        return json_data

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse LLM response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
