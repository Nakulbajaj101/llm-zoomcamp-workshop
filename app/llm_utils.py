from openai import OpenAI
from elasticsearch import Elasticsearch


INDEX_NAME = "course-questions"
CONTEXT_TEMPLATE = """
Section: {section}
Question: {question}
Answer: {text}
""".strip()

PROMPT_TEMPLATE = """
You're a course teaching assistant.
Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database.
Don't use other information outside of the provided CONTEXT.  

QUESTION: {user_question}

CONTEXT:

{context}
""".strip()

class Retrieval:
    def __init__(self, index_name: str):
        self.index_name = index_name
        self._host = "http://localhost:9200"
        self.es = Elasticsearch(self._host)
        

    def retrieve_documents(self, query: str, max_results: int=5) -> dict:
        
        search_query = {
            "size": max_results,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["question^3", "text", "section"],
                            "type": "best_fields"
                        }
                    },
                    "filter": {
                        "term": {
                            "course": "data-engineering-zoomcamp"
                        }
                    }
                }
            }
        }
        
        response = self.es.search(index=self.index_name, body=search_query)
        documents = [hit['_source'] for hit in response['hits']['hits']]
        return documents

class OpenAIRetrieval:
    def __init__(self, model_name: str= "gpt-3.5-turbo", index_name: str="course-questions"):
        self.model_name = model_name
        self.index_name = index_name
        self.client = OpenAI()
        self.ret = Retrieval(index_name=self.index_name)
        

    def build_context(self, documents: dict={}, context_template: str="") -> str:
        
        context_result = ""
        for doc in documents:
            doc_str = context_template.format(**doc)
            context_result += ("\n\n" + doc_str)

        return context_result.strip()

    def build_prompt(self,
                     user_question: str, 
                     documents: dict={},
                     context_template: str="",
                     prompt_template:str="") -> str:
        
        context = self.build_context(documents, context_template=context_template)
        prompt = prompt_template.format(
            user_question=user_question,
            context=context
        )
        
        return prompt

    def ask_openai(self, prompt: str) -> str:
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        return answer

    def qa_bot(self,
               user_question: str,
               context_template: str=CONTEXT_TEMPLATE,
               prompt_template: str=PROMPT_TEMPLATE) -> str:

        context_docs = self.ret.retrieve_documents(query=user_question)
        prompt = self.build_prompt(user_question, context_docs, context_template, prompt_template)
        answer = self.ask_openai(prompt)
        return answer

if __name__ == "__main__":
    oar = OpenAIRetrieval()
    answer = oar.qa_bot("I'm getting invalid reference format: repository name must be lowercase")
    print(answer)
            