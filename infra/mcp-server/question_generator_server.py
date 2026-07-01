import os
import json
import logging
import time
from dotenv import load_dotenv
from jinja2 import Template, Environment, select_autoescape
from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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

llm = os.getenv("LLM_PROVIDER", "gemini")
llm_instance = get_llm_instance(llm_provider=llm)


@mcp.tool(
    name="generate_exam_questions",
    description="Generate multiple choice questions based on provided content. Returns a JSON array of questions with options, correct answer, and explanations."
)
async def generate_exam_questions(
    context: str, 
    num_questions: int = 5
) -> list[dict]:
    """
    Generate multiple choice questions based on provided content.
    
    Args:
        context: The textual content to base the questions on.
        num_questions: The number of questions to generate (default is 5).

    Returns:
        A list of dictionaries, each representing a question with its options, correct answer, and explanation.
    """

    logger.info("Starting execution of `generate_exam_questions` MCP Tool")

    try:
        prompt = """
Based on the following content, generate up to {num_questions} multiple choice questions to test knowledge.
Respond ONLY with a valid JSON array, no markdown, no explanation.

The final result must follow this exact structure:
[
    {{
        "question": "Question text here",
        "options": [
            {{"id":"A","text":"option a"}},
            {{"id":"B","text":"option b"}},
            {{"id":"C","text":"option c"}},
            {{"id":"D","text":"option d"}},
        ],
        "correct": "A",
        "explanation": "Brief explanation of why this is correct"
    }}
]

Content:
{context}
        """.format(
            num_questions=num_questions,
            context=context
        )

        logger.info("Invoking LLM with prompt for question generation...")
        start = time.perf_counter()
        response = llm_instance.invoke(prompt)
        logger.info(f"LLM response received for question generation.\nQuestion generation time: {time.perf_counter() - start:.2f} seconds")

        questions =  json.loads(response.content)

        logger.info(f"Questions successfully generated.\nQuestions:\n{questions}\n")

        return questions

    except Exception as e:
        err_msg = f"Error generating exam questions. Details: {e}"
        logger.error(err_msg)
        return err_msg


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

    logger.info("Starting execution of `generate_exam_html` MCP Tool")

    try:
        template_str = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>

<style>
:root {
    --correct-bg: #d4edda;
    --correct-border: #28a745;
    --wrong-bg: #f8d7da;
    --wrong-border: #dc3545;
    --card-bg: #ffffff;
    --border: #dddddd;
    --hover: #f5f5f5;
}

body {
    font-family: Arial, Helvetica, sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
    background: #fafafa;
    color: #222;
}

.score-bar {
    position: sticky;
    top: 0;
    z-index: 100;
    background: white;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    font-weight: bold;
}

h1 {
    margin: 24px 0;
}

.question-block {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
}

.question-text {
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 16px;
}

.option {
    width: 100%;
    text-align: left;
    display: block;
    margin: 8px 0;
    padding: 12px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: white;
    cursor: pointer;
    transition: background-color 0.2s;
}

.option:hover:not(:disabled) {
    background: var(--hover);
}

.option:disabled {
    cursor: default;
}

.option.correct {
    background: var(--correct-bg);
    border-color: var(--correct-border);
}

.option.wrong {
    background: var(--wrong-bg);
    border-color: var(--wrong-border);
}

.explanation {
    display: none;
    margin-top: 15px;
    padding: 12px;
    border-left: 4px solid #ffc107;
    background: #fff8dc;
    border-radius: 4px;
}

.summary {
    display: none;
    margin-top: 30px;
    padding: 20px;
    border-radius: 8px;
    background: #eef6ff;
    border: 1px solid #bcdcff;
    text-align: center;
    font-size: 1.1rem;
    font-weight: bold;
}
</style>
</head>

<body>

<div class="score-bar">
    Score:
    <span id="score">0</span>
    /
    {{ questions|length }}
</div>

<h1>{{ title }}</h1>

{% for q in questions %}
{% set question_index = loop.index %}

<div
    class="question-block"
    id="question-{{ question_index }}"
    data-correct="{{ q.correct }}"
>

    <div class="question-text">
        {{ question_index }}. {{ q.question }}
    </div>

    {% for option in q.options %}
    <button
        class="option"
        data-question="{{ question_index }}"
        data-choice="{{ option.id }}"
    >
        <strong>{{ option.id }})</strong>
        {{ option.text }}
    </button>
    {% endfor %}

    <div
        class="explanation"
        id="explanation-{{ question_index }}"
    >
        💡 {{ q.explanation }}
    </div>

</div>
{% endfor %}

<div class="summary" id="summary"></div>

<script>
(() => {

    let score = 0;
    let answeredCount = 0;
    const totalQuestions = {{ questions|length }};

    const scoreElement = document.getElementById("score");

    document.querySelectorAll(".option").forEach(button => {

        button.addEventListener("click", () => {

            const questionId = button.dataset.question;

            const questionBlock = document.getElementById(
                `question-${questionId}`
            );

            if (questionBlock.dataset.answered === "true") {
                return;
            }

            questionBlock.dataset.answered = "true";

            const chosen = button.dataset.choice;
            const correct = questionBlock.dataset.correct;

            const buttons =
                questionBlock.querySelectorAll(".option");

            buttons.forEach(btn => {
                btn.disabled = true;

                if (btn.dataset.choice === correct) {
                    btn.classList.add("correct");
                }
            });

            if (chosen === correct) {
                score++;
            } else {
                button.classList.add("wrong");
            }

            answeredCount++;

            scoreElement.textContent = score;

            const explanation =
                document.getElementById(
                    `explanation-${questionId}`
                );

            explanation.style.display = "block";

            if (answeredCount === totalQuestions) {

                const summary =
                    document.getElementById("summary");

                summary.style.display = "block";

                summary.innerHTML =
                    `Finished!<br><br>` +
                    `Final Score: ${score} / ${totalQuestions}` +
                    ` (${Math.round(score / totalQuestions * 100)}%)`;

                summary.scrollIntoView({
                    behavior: "smooth"
                });
            }

        });

    });

})();
</script>

</body>
</html>"""
        env = Environment(
            autoescape=select_autoescape(
                enabled_extensions=("html", "xml"),
                default=True
            )
        )

        template = env.from_string(template_str)
        html = template.render(
            title=title,
            questions=questions
        )
        logger.info("HTML successfully generated")
        logger.info(f"\n{html}\n")

        return html
    except Exception as e:
        err_msg = f"Error generating HTML. Details: {e}"
        logger.error(err_msg)
        return err_msg



if __name__ == "__main__":
    logger.info("Starting MCP server...")
    app = mcp.streamable_http_app()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
