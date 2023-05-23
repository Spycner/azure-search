import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from approaches.approach import Approach
from text import nonewlines


# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
# (answer) with that prompt.
class ChatReadRetrieveReadApproach(Approach):
    prompt_prefix = """"<|im_start|>system
    Der Assistent hilft den Mitarbeitern des Unternehmens bei Fragen zu ihrem Gesundheitsplan und Fragen zum Mitarbeiterhandbuch. Seien Sie in Ihren Antworten kurz.
    Antworten Sie NUR mit den Fakten, die in der Liste der untenstehenden Quellen aufgeführt sind. Wenn nicht genügend Informationen vorhanden sind, sagen Sie, dass Sie es nicht wissen. Erzeugen Sie keine Antworten, die nicht die untenstehenden Quellen verwenden. Wenn eine klärende Frage an den Benutzer hilfreich wäre, stellen Sie die Frage.
    Geben Sie tabellarische Informationen als HTML-Tabelle zurück. Geben Sie kein Markdown-Format zurück.
    Jede Quelle hat einen Namen, gefolgt von einem Doppelpunkt und den tatsächlichen Informationen, geben Sie immer den Quellennamen für jede Tatsache, die Sie in der Antwort verwenden, an. Verwenden Sie eckige Klammern, um die Quelle zu referenzieren, z.B. [info1.txt]. Kombinieren Sie keine Quellen, listen Sie jede Quelle separat auf, z.B. [info1.txt][info2.pdf].
    {follow_up_questions_prompt}
    {injected_prompt}
    Quellen:
    {sources}
    <|im_end|>
    {chat_history}
    """

    follow_up_questions_prompt_content = """Erzeugen Sie drei sehr kurze Folgefragen, die der Benutzer wahrscheinlich als nächstes zu seinem Gesundheitsplan und Mitarbeiterhandbuch stellen würde.
    Verwenden Sie doppelte spitze Klammern, um auf die Fragen zu verweisen, z.B. <<Gibt es Ausschlüsse für Rezepte?>>.
    Versuchen Sie, Fragen, die bereits gestellt wurden, nicht zu wiederholen.
    Generieren Sie nur Fragen und keinen Text vor oder nach den Fragen, wie z.B. 'Nächste Fragen'"""

    query_prompt_template = """Im Folgenden finden Sie eine Geschichte des bisherigen Gesprächs und eine neue Frage des Benutzers, die durch die Suche in einer Wissensdatenbank über Gesundheitspläne für Mitarbeiter und das Mitarbeiterhandbuch beantwortet werden muss.
    Erzeugen Sie eine Suchanfrage basierend auf dem Gespräch und der neuen Frage.
    Fügen Sie keine zitierten Quellendateinamen und Dokumentennamen, wie z.B. info.txt oder doc.pdf, in die Suchanfragenterme ein.
    Fügen Sie keinen Text in [] oder <<>> in die Suchanfragenterme ein.
    Wenn die Frage nicht auf Englisch ist, übersetzen Sie die Frage ins Englische, bevor Sie die Suchanfrage generieren.

    Chatverlauf:
    {chat_history}

    Frage:
    {question}

    Suchanfrage:
    """

    def __init__(
        self,
        search_client: SearchClient,
        chatgpt_deployment: str,
        gpt_deployment: str,
        sourcepage_field: str,
        content_field: str,
    ):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.gpt_deployment = gpt_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def run(self, history: list[dict], overrides: dict) -> any:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = (
            "category ne '{}'".format(exclude_category.replace("'", "''"))
            if exclude_category
            else None
        )

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        prompt = self.query_prompt_template.format(
            chat_history=self.get_chat_history_as_text(
                history, include_last_turn=False
            ),
            question=history[-1]["user"],
        )
        completion = openai.Completion.create(
            engine=self.gpt_deployment,
            prompt=prompt,
            temperature=0.0,
            max_tokens=32,
            n=1,
            stop=["\n"],
        )
        q = completion.choices[0].text

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query
        if overrides.get("semantic_ranker"):
            r = self.search_client.search(
                q,
                filter=filter,
                query_type=QueryType.SEMANTIC,
                query_language="de-de",
                query_speller="lexicon",
                semantic_configuration_name="default",
                top=top,
                query_caption="extractive|highlight-false"
                if use_semantic_captions
                else None,
            )
        else:
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            results = [
                doc[self.sourcepage_field]
                + ": "
                + nonewlines(" . ".join([c.text for c in doc["@search.captions"]]))
                for doc in r
            ]
        else:
            results = [
                doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field])
                for doc in r
            ]
        content = "\n".join(results)

        follow_up_questions_prompt = (
            self.follow_up_questions_prompt_content
            if overrides.get("suggest_followup_questions")
            else ""
        )

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        if prompt_override is None:
            prompt = self.prompt_prefix.format(
                injected_prompt="",
                sources=content,
                chat_history=self.get_chat_history_as_text(history),
                follow_up_questions_prompt=follow_up_questions_prompt,
            )
        elif prompt_override.startswith(">>>"):
            prompt = self.prompt_prefix.format(
                injected_prompt=prompt_override[3:] + "\n",
                sources=content,
                chat_history=self.get_chat_history_as_text(history),
                follow_up_questions_prompt=follow_up_questions_prompt,
            )
        else:
            prompt = prompt_override.format(
                sources=content,
                chat_history=self.get_chat_history_as_text(history),
                follow_up_questions_prompt=follow_up_questions_prompt,
            )

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history
        completion = openai.Completion.create(
            engine=self.chatgpt_deployment,
            prompt=prompt,
            temperature=overrides.get("temperature") or 0.7,
            max_tokens=1024,
            n=1,
            stop=["<|im_end|>", "<|im_start|>"],
        )

        return {
            "data_points": results,
            "answer": completion.choices[0].text,
            "thoughts": f"Searched for:<br>{q}<br><br>Prompt:<br>"
            + prompt.replace("\n", "<br>"),
        }

    def get_chat_history_as_text(
        self, history, include_last_turn=True, approx_max_tokens=1000
    ) -> str:
        history_text = ""
        for h in reversed(history if include_last_turn else history[:-1]):
            history_text = (
                """<|im_start|>user"""
                + "\n"
                + h["user"]
                + "\n"
                + """<|im_end|>"""
                + "\n"
                + """<|im_start|>assistant"""
                + "\n"
                + (h.get("bot") + """<|im_end|>""" if h.get("bot") else "")
                + "\n"
                + history_text
            )
            if len(history_text) > approx_max_tokens * 4:
                break
        return history_text
