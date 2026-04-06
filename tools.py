from langchain_core.tools import tool
import math
import datetime
import requests
import urllib.parse
import certifi

# Explicit SSL + timeout defaults for all outgoing requests
_SSL  = certifi.where()
_TIMEOUT = 10
_HEADERS = {"User-Agent": "PersonalAIAssistant/1.0 (educational project)"}


def make_search_document_tool(retriever):
    """
    Factory — creates a search_document tool bound to the current retriever.
    Called every time the user uploads a new document.
    """
    @tool
    def search_document(query: str) -> str:
        """
        Searches the uploaded document for information relevant to the query.
        Use this whenever the user asks about content from their document.
        """
        if retriever is None:
            return "No document has been uploaded yet."
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant content found in the document for that query."
        parts = []
        for i, doc in enumerate(docs):
            page = doc.metadata.get("page", "?")
            parts.append(f"[Chunk {i+1} | page {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    return search_document


@tool
def calculator(expression: str) -> str:
    """
    Calculates mathematical expressions.
    Examples: '15% * 8500' or '1500 * 12'
    """
    try:
        expression = expression.replace('%', '/100')
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"Result: {round(result, 2)}"
    except Exception as e:
        return f"Could not calculate: {str(e)}"


@tool
def word_counter(text: str) -> str:
    """
    Counts the number of words and characters in a given text.
    Useful for checking document length.
    """
    words = len(text.split())
    chars = len(text)
    return f"Words: {words} | Characters: {chars}"


@tool
def summarize_request(topic: str) -> str:
    """
    Returns an instruction to summarize a specific topic from the document.
    Use this when the user asks to summarize a section or topic.
    """
    return f"Please summarize all information related to: {topic}"


@tool
def get_current_date(dummy: str = "") -> str:
    """
    Returns the current date and time.
    Use this when the user asks what day or time it is.
    """
    now = datetime.datetime.now()
    return f"Current date and time: {now.strftime('%d/%m/%Y %H:%M')}"


@tool
def compare_numbers(input_str: str) -> str:
    """
    Compares two numbers and returns which is larger, smaller, or if equal.
    Input format: 'number1,number2'. Example: '5000,8500'
    """
    try:
        parts = input_str.split(',')
        a, b = float(parts[0].strip()), float(parts[1].strip())
        if a > b:
            return f"{a} is larger than {b} (difference: {round(a - b, 2)})"
        elif b > a:
            return f"{b} is larger than {a} (difference: {round(b - a, 2)})"
        else:
            return "The numbers are equal"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def bullet_list_formatter(text: str) -> str:
    """
    Formats a block of text into a clean bullet point list.
    Use this when the user asks to format or organize information as bullet points.
    """
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    bullets = '\n'.join([f"• {s}" for s in sentences])
    return f"Formatted list:\n{bullets}"


@tool
def keyword_extractor(text: str) -> str:
    """
    Extracts the most important keywords from a given text.
    Use when the user wants to find key terms or topics in a passage.
    """
    stopwords = {
        'the','a','an','is','in','on','at','of','and','or','to','for',
        'with','that','this','it','be','are','was','were','has','have',
        'from','by','as','but','not','so','if','do','did','will','can',
        'i','you','he','she','we','they','my','your','his','her','our'
    }
    words = text.lower().split()
    words = [w.strip('.,!?()[]":;') for w in words]
    freq = {}
    for w in words:
        if w and w not in stopwords and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:8]
    return "Keywords: " + ', '.join([w for w, _ in top])


@tool
def get_weather(city: str) -> str:
    """
    Gets the current weather for a given city.
    Use when the user asks about weather conditions anywhere in the world.
    Example: 'Tel Aviv' or 'London'
    """
    try:
        url = f"https://wttr.in/{city}?format=j1"
        res = requests.get(url, timeout=_TIMEOUT, verify=_SSL, headers=_HEADERS).json()
        current = res['current_condition'][0]
        temp = current['temp_C']
        feels = current['FeelsLikeC']
        desc = current['weatherDesc'][0]['value']
        humidity = current['humidity']
        return (
            f"Weather in {city}:\n"
            f"Temperature: {temp}°C (feels like {feels}°C)\n"
            f"Condition: {desc}\n"
            f"Humidity: {humidity}%"
        )
    except Exception as e:
        return f"Could not fetch weather for {city}: {str(e)}"


@tool
def get_crypto_price(coin: str) -> str:
    """
    Gets the current price of a cryptocurrency in USD.
    Use when the user asks about crypto prices.
    Examples: 'bitcoin', 'ethereum', 'solana'
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": coin.lower(), "vs_currencies": "usd", "include_24hr_change": "true"}
        res = requests.get(url, params=params, timeout=_TIMEOUT, verify=_SSL, headers=_HEADERS).json()
        data = res[coin.lower()]
        price = data['usd']
        change = round(data['usd_24h_change'], 2)
        direction = "up" if change > 0 else "down"
        return (
            f"{coin.capitalize()} price:\n"
            f"${price:,}\n"
            f"24h change: {direction} {abs(change)}%"
        )
    except Exception as e:
        return f"Could not fetch price for {coin}: {str(e)}"


@tool
def search_wikipedia(query: str) -> str:
    """
    Searches Wikipedia and returns a short summary on any topic.
    Use when the user asks about a concept, person, place, or event.
    Example: 'Albert Einstein', 'Machine Learning', 'Prime Minister of Israel'
    Always write the query in English for best results.
    """
    try:
        # Step 1: search for the best matching page title
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1,
        }
        search_res = requests.get(search_url, params=search_params, headers=_HEADERS, timeout=_TIMEOUT, verify=_SSL).json()
        results = search_res.get("query", {}).get("search", [])
        if not results:
            return f"No Wikipedia results found for: {query}"

        best_title = results[0]["title"]

        # Step 2: fetch summary for that title
        encoded_title = urllib.parse.quote(best_title.replace(" ", "_"), safe="")
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
        res = requests.get(summary_url, headers=_HEADERS, timeout=_TIMEOUT, verify=_SSL).json()
        title = res.get("title", best_title)
        extract = res.get("extract", "No summary found.")
        sentences = extract.split(". ")[:3]
        summary = ". ".join(sentences)
        return f"Wikipedia — {title}:\n{summary}"
    except Exception as e:
        return f"Could not search Wikipedia for '{query}': {str(e)}"


@tool
def get_exchange_rate(pair: str) -> str:
    """
    Gets the current exchange rate between two currencies.
    Input format: 'FROM,TO'. Examples: 'USD,ILS' or 'EUR,USD'
    """
    try:
        parts = pair.upper().split(',')
        from_cur, to_cur = parts[0].strip(), parts[1].strip()
        url = f"https://open.er-api.com/v6/latest/{from_cur}"
        res = requests.get(url, timeout=_TIMEOUT, verify=_SSL, headers=_HEADERS).json()
        rate = res['rates'][to_cur]
        return f"1 {from_cur} = {round(rate, 4)} {to_cur}"
    except Exception as e:
        return f"Could not fetch exchange rate: {str(e)}"


tools = [
    calculator,
    word_counter,
    summarize_request,
    get_current_date,
    compare_numbers,
    bullet_list_formatter,
    keyword_extractor,
    get_weather,
    get_crypto_price,
    search_wikipedia,
    get_exchange_rate,
]