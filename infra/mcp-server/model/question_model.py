from pydantic import BaseModel
from typing import Literal

class Option(BaseModel):
    id: Literal["A", "B", "C", "D"]
    text: str

class Question(BaseModel):
    question: str
    options: list[Option]
    correct: Literal["A", "B", "C", "D"]
    explanation: str

class QuestionList(BaseModel):
    questions: list[Question]