import openai
from approaches.approach import Approach
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from langchain.llms.openai import AzureOpenAI
from langchain.prompts import PromptTemplate, BasePromptTemplate
from langchain.callbacks.base import CallbackManager
from langchain.agents import Tool, AgentExecutor
from langchain.agents.react.base import ReActDocstoreAgent
from langchainadapters import HtmlCallbackHandler
from text import nonewlines
from typing import List


class ReadDecomposeAsk(Approach):
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

    def search(self, q: str, overrides: dict) -> str:
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
            self.results = [
                doc[self.sourcepage_field]
                + ":"
                + nonewlines(" . ".join([c.text for c in doc["@search.captions"]]))
                for doc in r
            ]
        else:
            self.results = [
                doc[self.sourcepage_field]
                + ":"
                + nonewlines(doc[self.content_field][:500])
                for doc in r
            ]
        return "\n".join(self.results)

    def lookup(self, q: str) -> str:
        r = self.search_client.search(
            q,
            top=1,
            include_total_count=True,
            query_type=QueryType.SEMANTIC,
            query_language="de-de",
            query_speller="lexicon",
            semantic_configuration_name="default",
            query_answer="extractive|count-1",
            query_caption="extractive|highlight-false",
        )

        answers = r.get_answers()
        if answers and len(answers) > 0:
            return answers[0].text
        if r.get_count() > 0:
            return "\n".join(d["content"] for d in r)
        return None

    def run(self, q: str, overrides: dict) -> any:
        # Not great to keep this as instance state, won't work with interleaving (e.g. if using async), but keeps the example simple
        self.results = None

        # Use to capture thought process during iterations
        cb_handler = HtmlCallbackHandler()
        cb_manager = CallbackManager(handlers=[cb_handler])

        llm = AzureOpenAI(
            deployment_name=self.openai_deployment,
            temperature=overrides.get("temperature") or 0.3,
            openai_api_key=openai.api_key,
        )
        tools = [
            Tool(name="Search", func=lambda q: self.search(q, overrides)),
            Tool(name="Lookup", func=self.lookup),
        ]

        # Like results above, not great to keep this as a global, will interfere with interleaving
        global prompt
        prompt_prefix = overrides.get("prompt_template")
        prompt = PromptTemplate.from_examples(
            EXAMPLES,
            SUFFIX,
            ["input", "agent_scratchpad"],
            prompt_prefix + "\n\n" + PREFIX if prompt_prefix else PREFIX,
        )

        agent = ReAct.from_llm_and_tools(llm, tools)
        chain = AgentExecutor.from_agent_and_tools(
            agent, tools, verbose=True, callback_manager=cb_manager
        )
        result = chain.run(q)

        # Fix up references to they look like what the frontend expects ([] instead of ()), need a better citation format since parentheses are so common
        result = result.replace("(", "[").replace(")", "]")

        return {
            "data_points": self.results or [],
            "answer": result,
            "thoughts": cb_handler.get_and_reset_log(),
        }


class ReAct(ReActDocstoreAgent):
    @classmethod
    def create_prompt(cls, tools: List[Tool]) -> BasePromptTemplate:
        return prompt


# Modifizierte Version des ReAct-Prompts von langchain, der Anweisungen und Beispiele für die Zitierung von Informationsquellen enthält
# Modifizierte Version des ReAct-Prompts von langchain, der Anweisungen und Beispiele für die Zitierung von Informationsquellen enthält
EXAMPLES = [
    """Frage: Wie hoch ist der Höhenbereich für das Gebiet, in das der östliche Sektor der
Colorado Orogenese hineinreicht?
Gedanke 1: Ich muss die Colorado Orogenese suchen, das Gebiet finden, in das der östliche Sektor
der Colorado Orogenese hineinreicht, dann den Höhenbereich des
Gebiets finden.
Aktion 1: Suche[Colorado Orogenese]
Beobachtung 1: [info1.pdf] Die Colorado Orogenese war eine Episode der Gebirgsbildung (eine Orogenese) in
Colorado und den umliegenden Gebieten.
Gedanke 2: Es wird nicht der östliche Sektor erwähnt. Also muss ich den östlichen Sektor suchen.
Aktion 2: Suche[östlicher Sektor]
Beobachtung 2: [info2.txt] (Ergebnis 1 / 1) Der östliche Sektor erstreckt sich in die High Plains und wird
als Central Plains Orogenese bezeichnet.
Gedanke 3: Der östliche Sektor der Colorado Orogenese erstreckt sich in die High Plains. Also muss ich
die High Plains suchen und deren Höhenbereich finden.
Aktion 3: Suche[High Plains]
Beobachtung 3: [some_file.pdf] High Plains bezieht sich auf eine von zwei unterschiedlichen Landregionen
Gedanke 4: Ich muss stattdessen High Plains (Vereinigte Staaten) suchen.
Aktion 4: Suche[High Plains (Vereinigte Staaten)]
Beobachtung 4: [filea.pdf] Die High Plains sind eine Teilregion der Great Plains. [another-ref.docx] Von Ost nach West, die
High Plains steigen in der Höhe von etwa 1.800 bis 7.000 ft (550 bis 2.130
m).
Gedanke 5: Die High Plains steigen in der Höhe von etwa 1.800 bis 7.000 ft, also ist die Antwort
1.800 bis 7.000 ft.
Aktion 5: Fertig[1.800 bis 7.000 ft (filea.pdf) ]""",
    """Frage: Die Musikerin und Satirikerin Allie Goertz hat ein Lied über die "Die Simpsons"
Charakter Milhouse geschrieben, den Matt Groening nach wem benannt hat?
Gedanke 1: Die Frage vereinfacht sich zu "Der Simpsons" Charakter Milhouse ist benannt nach
wem. Ich muss nur Milhouse suchen und finden, nach wem es benannt ist.
Aktion 1: Suche[Milhouse]
Beobachtung 1: [info7.pdf] Milhouse Mussolini Van Houten ist eine wiederkehrende Figur in der animierten
Fernsehserie The Simpsons, gesprochen von Pamela Hayden und kreiert von Matt
Groening.
Gedanke 2: Der Absatz sagt nicht, nach wem Milhouse benannt ist, vielleicht kann ich
"benannt nach" suchen.
Aktion 2: Suche[benannt nach]
Beobachtung 2: [historyref2.txt] (Ergebnis 1 / 1) Milhouse wurde nach US-Präsident Richard Nixon benannt, dessen
zweiter Vorname Milhous war.
Gedanke 3: Milhouse wurde nach US-Präsident Richard Nixon benannt, also ist die Antwort
Richard Nixon.
Aktion 3: Fertig[Richard Nixon (historyref2.txt) ]""",
    """Frage: Welche Dokumentation handelt von finnischen Rockgruppen, Adam Clayton Powell oder The
Saimaa Gesture?
Gedanke 1: Ich muss Adam Clayton Powell und The Saimaa Gesture suchen und herausfinden, welche
Dokumentation über finnische Rockgruppen handelt.
Aktion 1: Suche[Adam Clayton Powell]
Beobachtung 1: [info9.pdf] Konnte [Adam Clayton Powell] nicht finden. Ähnlich: ['Adam Clayton Powell
III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office
Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell
(film)', 'Giancarlo Esposito'].
Gedanke 2: Um die Dokumentation zu finden, kann ich Adam Clayton Powell (film) suchen.
Aktion 2: Suche[Adam Clayton Powell (film)]
Beobachtung 2: [data123.txt] Adam Clayton Powell ist ein amerikanischer Dokumentarfilm von 1989, Regie von
Richard Kilberg. Der Film handelt vom Aufstieg und Fall des einflussreichen
afroamerikanischen Politikers Adam Clayton Powell Jr.[3][4] Er wurde später als Teil der PBS-Serie The American Experience ausgestrahlt.
Gedanke 3: Adam Clayton Powell (film) ist eine Dokumentation über einen afroamerikanischen
Politiker, nicht über finnische Rockgruppen. So muss die Dokumentation über finnische Rockgruppen
stattdessen The Saimaa Gesture sein.
Aktion 3: Fertig[The Saimaa Gesture (data123.txt) ]""",
    """Frage: Welchen Beruf haben Nicholas Ray und Elia Kazan gemeinsam?
Gedanke 1: Ich muss Nicholas Ray und Elia Kazan suchen, ihre Berufe finden und dann
den Beruf finden, den sie gemeinsam haben.
Aktion 1: Suche[Nicholas Ray]
Beobachtung 1: [files-987.png] Nicholas Ray (geboren als Raymond Nicholas Kienzle Jr., 7. August 1911 - 16.
Juni 1979) war ein amerikanischer Filmregisseur, Drehbuchautor und Schauspieler, am besten bekannt für
den Film Rebel Without a Cause aus dem Jahr 1955.
Gedanke 2: Berufe von Nicholas Ray sind Regisseur, Drehbuchautor und Schauspieler. Ich muss
als nächstes Elia Kazan suchen und seine Berufe finden.
Aktion 2: Suche[Elia Kazan]
Beobachtung 2: [files-654.txt] Elia Kazan war ein amerikanischer Film- und Theaterregisseur, Produzent, Drehbuchautor
und Schauspieler.
Gedanke 3: Berufe von Elia Kazan sind Regisseur, Produzent, Drehbuchautor und Schauspieler.
So haben Nicholas Ray und Elia Kazan den Beruf Regisseur,
Drehbuchautor und Schauspieler gemeinsam.
Aktion 3: Fertig[Regisseur, Drehbuchautor, Schauspieler (files-987.png)(files-654.txt) ]""",
    """Frage: Welches Magazin wurde zuerst gestartet, Arthur's Magazine oder First for Women?
Gedanke 1: Ich muss Arthur's Magazine und First for Women suchen und herausfinden, welches zuerst
gestartet wurde.
Aktion 1: Suche[Arthur's Magazine]
Beobachtung 1: [magazines-1850.pdf] Arthur's Magazine (1844-1846) war eine amerikanische literarische Zeitschrift, die
im 19. Jahrhundert in Philadelphia veröffentlicht wurde.
Gedanke 2: Arthur's Magazine wurde 1844 gestartet. Als nächstes muss ich First for Women
suchen.
Aktion 2: Suche[First for Women]
Beobachtung 2: [magazines-1900.pdf] First for Women ist eine Frauenzeitschrift, die von Bauer Media Group in den
USA veröffentlicht wird.[1] Das Magazin wurde 1989 gestartet.
Gedanke 3: First for Women wurde 1989 gestartet. 1844 (Arthur's Magazine) < 1989 (First
for Women), also wurde Arthur's Magazine zuerst gestartet.
Aktion 3: Fertig[Arthur's Magazine (magazines-1850.pdf)(magazines-1900.pdf) ]""",
    """Frage: Waren Pavel Urysohn und Leonid Levin für die gleiche Art von Arbeit bekannt?
Gedanke 1: Ich muss Pavel Urysohn und Leonid Levin suchen, ihre Arten von Arbeit finden,
und herausfinden, ob sie die gleichen sind.
Aktion 1: Suche[Pavel Urysohn]
Beobachtung 1: [info4444.pdf] Pavel Samuilovich Urysohn (3. Februar 1898 - 17. August 1924) war ein sowjetischer
Mathematiker, der am besten für seine Beiträge zur Dimensionstheorie bekannt ist.
Gedanke 2: Pavel Urysohn ist ein Mathematiker. Als nächstes muss ich Leonid Levin suchen und
seine Art von Arbeit finden.
Aktion 2: Suche[Leonid Levin]
Beobachtung 2: [datapoints_aaa.txt] Leonid Anatolievich Levin ist ein sowjetisch-amerikanischer Mathematiker und Informatiker.
Gedanke 3: Leonid Levin ist ein Mathematiker und Informatiker. So sind Pavel Urysohn
und Leonid Levin für die gleiche Art von Arbeit bekannt.
Aktion 3: Fertig[Ja (info4444.pdf)(datapoints_aaa.txt) ]""",
]
SUFFIX = """\nFrage: {input}
{agent_scratchpad}"""
PREFIX = (
    "Beantworten Sie Fragen wie in den folgenden Beispielen gezeigt, indem Sie die Frage in einzelne Such- oder Nachschlageaktionen aufteilen, um Fakten zu finden, bis Sie die Frage beantworten können. "
    "Beobachtungen werden durch ihren Quellennamen in eckigen Klammern gekennzeichnet, Quellennamen MÜSSEN in den Antworten bei den Aktionen enthalten sein."
    "Beantworten Sie die Fragen nur mit Informationen aus Beobachtungen, spekulieren Sie nicht."
)
