from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langfuse.callback import CallbackHandler
import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

load_dotenv()

chat = ChatCohere()
app = FastAPI()

langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


class Item(BaseModel):
    content: str


@app.post("/summary/")
async def create_item(item: Item):
    prompt = f"provide a summary of following email {item.content}"
    messages = [HumanMessage(content=prompt)]
    return chat.invoke(messages, config={"callbacks": [langfuse_handler]})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
