import openai
from approaches.approach import Approach
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from text import nonewlines


# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
# (answer) with that prompt.
class RetrieveThenReadApproach(Approach):
    template = (
        "Sie sind ein intelligenter Assistent, der den Mitarbeitern von Contoso Inc bei ihren Fragen zum Gesundheitsplan und zum Mitarbeiterhandbuch hilft. "
        + "Verwenden Sie 'Sie' um sich auf die Person zu beziehen, die die Fragen stellt, auch wenn diese mit 'Ich' fragt. "
        + "Beantworten Sie die folgende Frage ausschließlich mit den Daten aus den untenstehenden Quellen. "
        + "Geben Sie tabellarische Informationen als HTML-Tabelle zurück. Verwenden Sie kein Markdown-Format. "
        + "Jede Quelle hat einen Namen, gefolgt von einem Doppelpunkt und den eigentlichen Informationen, führen Sie immer den Quellennamen für jede Tatsache, die Sie in der Antwort verwenden, auf. "
        + "Wenn Sie die Frage nicht mit den untenstehenden Quellen beantworten können, sagen Sie, dass Sie es nicht wissen. "
        + """

###
Frage: 'Was ist die Selbstbeteiligung für den Mitarbeiterplan für einen Besuch in Overlake in Bellevue?'

Quellen:
info1.txt: Die Selbstbeteiligung hängt davon ab, ob Sie sich innerhalb oder außerhalb des Netzwerks befinden. Die Selbstbeteiligung innerhalb des Netzwerks beträgt $500 für Mitarbeiter und $1000 für Familien. Die Selbstbeteiligung außerhalb des Netzwerks beträgt $1000 für Mitarbeiter und $2000 für Familien.
info2.pdf: Overlake gehört zum Netzwerk des Mitarbeiterplans.
info3.pdf: Overlake ist der Name des Gebiets, das einen Park-and-Ride-Platz in der Nähe von Bellevue umfasst.
info4.pdf: Zu den Institutionen innerhalb des Netzwerks gehören Overlake, Swedish und andere in der Region

Antwort:
Die Selbstbeteiligung innerhalb des Netzwerks beträgt $500 für Mitarbeiter und $1000 für Familien [info1.txt] und Overlake gehört zum Netzwerk des Mitarbeiterplans [info2.pdf][info4.pdf].

###
Frage: '{q}'?

Quellen:
{retrieved}

Antwort:
"""
    )

    def __init__(
        self,
        search_client: SearchClient,
        openai_deployment: str,
        sourcepage_field: str,
        content_field: str,
    ):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def run(self, q: str, overrides: dict) -> any:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = (
            "category ne '{}'".format(exclude_category.replace("'", "''"))
            if exclude_category
            else None
        )

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

        prompt = (overrides.get("prompt_template") or self.template).format(
            q=q, retrieved=content
        )
        completion = openai.Completion.create(
            engine=self.openai_deployment,
            prompt=prompt,
            temperature=overrides.get("temperature") or 0.3,
            max_tokens=1024,
            n=1,
            stop=["\n"],
        )

        return {
            "data_points": results,
            "answer": completion.choices[0].text,
            "thoughts": f"Question:<br>{q}<br><br>Prompt:<br>"
            + prompt.replace("\n", "<br>"),
        }
