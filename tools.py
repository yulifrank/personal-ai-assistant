from langchain_core.tools import tool
import math
import datetime
import requests


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
        res = requests.get(url, timeout=5).json()
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
        res = requests.get(url, params=params, timeout=5).json()
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
    Example: 'Albert Einstein' or 'Machine Learning'
    """
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")
        res = requests.get(url, timeout=5).json()
        title = res.get('title', query)
        extract = res.get('extract', 'No summary found.')
        sentences = extract.split('. ')[:3]
        summary = '. '.join(sentences)
        return f"Wikipedia — {title}:\n{summary}"
    except Exception as e:
        return f"Could not search Wikipedia for {query}: {str(e)}"


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
        res = requests.get(url, timeout=5).json()
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