import os
import json
import time
from dotenv import load_dotenv
from jinja2 import Template
from mcp.server.fastmcp import FastMCP

def get_api_key(key_name: str) -> str:
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f"API Key must be valid for env var {key_name}")
    return api_key


def get_llm_instance(llm_provider: str, temperature: float = 0.2):
    if llm_provider.lower() == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=temperature, 
            api_key=get_api_key("GROQ_API_KEY")
        )
    elif llm_provider.lower() == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4.1-mini", 
            temperature=temperature,
            api_key=get_api_key("OPENAI_API_KEY")
        )
    elif llm_provider.lower() == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=temperature,
            api_key=get_api_key("GEMINI_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported LLM: {llm_provider}")


# Initialize MCP server
mcp = FastMCP()

load_dotenv()

llm = os.getenv("LLM_PROVIDER", "groq")
llm_instance = get_llm_instance(llm_provider=llm)


@mcp.tool(
    name="generate_exam_questions",
    description="Generate multiple choice questions based on provided content. Returns a JSON array of questions with options, correct answer, and explanations."
)
async def generate_exam_questions(
    context: str, 
    num_questions: int = 10
) -> list[dict]:
    """
    Generate multiple choice questions based on provided content.
    
    Args:
        context: The textual content to base the questions on.
        num_questions: The number of questions to generate (default is 10).

    Returns:
        A list of dictionaries, each representing a question with its options, correct answer, and explanation.
    """

    prompt = """
    Based on the following content, generate up to {num_questions} multiple choice questions to test knowledge.
    Respond ONLY with a valid JSON array, no markdown, no explanation.

    The final result must follow this exact structure:
    [
        {{
            "question": "Question text here",
            "options": ["A) option", "B) option", "C) option", "D) option"],
            "answer": "A) correct option",
            "explanation": "Brief explanation of why this is correct"
        }}
    ]

    Content:
    {context}
    """.format(
        num_questions=num_questions,
        context=context
    )

    print("Invoking LLM with prompt for question generation...")
    start = time.perf_counter()
    response = llm_instance.invoke(prompt)
    print("LLM response received for question generation.")
    end = time.perf_counter()
    print(f"Question generation time: {end - start:.2f} seconds")
    return json.loads(response.content)


@mcp.tool(
    name="generate_exam_html",
    description="Convert a list of questions into a standalone interactive HTML exam. Input is a JSON array of questions with options, correct answer, and explanations."
)
def generate_exam_html(questions: list[dict], title: str = "Knowledge Check") -> str:
    """
    Convert a list of questions into a standalone interactive HTML exam.
    
    Args:
        questions: A list of dictionaries, each representing a question with its options, correct answer, and explanation.
        title: The title of the exam (default is "Knowledge Check").

    Returns:
        A string containing the HTML code for the interactive exam.
    """


    template_str = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{{ title }}</title>
        <style>
            /* all styles inline — no external dependencies */
            body { font-family: Georgia, serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #fafaf8; color: #1a1a1a; }
            h1 { font-size: 1.8rem; border-bottom: 2px solid #1a1a1a; padding-bottom: 12px; }
            .question-block { background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 24px; margin: 20px 0; }
            .question-text { font-size: 1.05rem; font-weight: bold; margin-bottom: 16px; }
            .option { display: block; padding: 10px 14px; margin: 6px 0; border: 1px solid #ddd; border-radius: 6px; cursor: pointer; transition: background 0.2s; }
            .option:hover { background: #f0f0f0; }
            .option.correct { background: #d4edda; border-color: #28a745; }
            .option.wrong { background: #f8d7da; border-color: #dc3545; }
            .explanation { display: none; margin-top: 14px; padding: 12px; background: #fff8dc; border-left: 4px solid #f0ad4e; border-radius: 4px; font-size: 0.9rem; }
            .score-bar { position: sticky; top: 0; background: white; border-bottom: 1px solid #ddd; padding: 12px 20px; font-weight: bold; z-index: 10; }
        </style>
    </head>
    <body>
        <div class="score-bar">Score: <span id="score">0</span> / {{ questions|length }}</div>
        <h1>{{ title }}</h1>

        {% for q in questions %}
        <div class="question-block" id="q{{ loop.index }}">
            <div class="question-text">{{ loop.index }}. {{ q.question }}</div>
            {% for option in q.options %}
            <button class="option"
                onclick="answer({{ loop.index0 }}, {{ loop.index }}, '{{ option }}', '{{ q.answer }}')"
                id="q{{ loop.index }}-opt{{ loop.index0 }}">
                {{ option }}
            </button>
            {% endfor %}
            <div class="explanation" id="exp{{ loop.index }}">
                💡 {{ q.explanation }}
            </div>
        </div>
        {% endfor %}

        <script>
            let score = 0;
            const answered = new Set();

            function answer(optIdx, qIdx, chosen, correct) {
                if (answered.has(qIdx)) return;
                answered.add(qIdx);

                const block = document.getElementById('q' + qIdx);
                block.querySelectorAll('.option').forEach(btn => btn.style.pointerEvents = 'none');

                const selected = document.getElementById('q' + qIdx + '-opt' + optIdx);
                selected.classList.add(chosen === correct ? 'correct' : 'wrong');

                if (chosen !== correct) {
                    block.querySelectorAll('.option').forEach(btn => {
                        if (btn.textContent.trim() === correct) btn.classList.add('correct');
                    });
                } else {
                    score++;
                    document.getElementById('score').textContent = score;
                }

                document.getElementById('exp' + qIdx).style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return Template(template_str).render(title=title, questions=questions)


if __name__ == "__main__":
    print("Starting MCP server...")
    app = mcp.streamable_http_app()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
#    mcp.run(
#        transport="streamable-http",
#        host="0.0.0.0",
#        port=8000,
#    )
