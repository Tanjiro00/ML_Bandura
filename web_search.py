from langchain_community.tools import DuckDuckGoSearchRun
def web_search(query):
    search_tool = DuckDuckGoSearchRun(max_results=5)
    results = search_tool.run('Кто является победителем котировочной сессии?')

    formatted_results = ""
    if isinstance(results, list):
        for res in results:
            title = res.get("title", "")
            snippet = res.get("body", "")
            formatted_results += f"{title}\n{snippet}\n\n"
    else:
        formatted_results = str(results)
    return formatted_results

