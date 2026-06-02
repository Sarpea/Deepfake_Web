import re
from collections import Counter
import unicodedata




def frequencies_analisys(keyword_doc):
    """
    Debe recibir una lista de objetos con la forma:
    [
    {"title": titulo, "keywords": keywords=[str,str,str,str...},
    {"title": titulo, "keywords": keywords=[str,str,str,str...},
    {"title": titulo, "keywords": keywords=[str,str,str,str...}
    ]

    Y devuelve una lista las palabras más frecuentes de titulos y keywords por separado, junto con la cantidad de veces que aparecen y la frecuencia (porcentaje de videos que contienen la palabra) con la forma
    {"palabra1":{"amount": cantidad, "percentage": porcentaje}, "palabra2":{"amount": cantidad, "percentage": porcentaje},...} eliminando aquellas palabras que solo aparecen una vez
    """
    titles = []
    keywords = []
    for obj in keyword_doc:
        titles.append(obj["title"])
        keywords.extend(obj["keywords"])

    return keyword_frequencies(titles, keywords)
    




def keyword_frequencies(title_list, keywords_list):

    """
    title_list: lista de todos los titulos de los videos a analizar
    keywords: lista de las keywords de cada video
    Devuelve una lista con las palabras ordenadas por frecuencia incluyendo cantidad de apariciones y el porcentaje de aparicion eliminando las que solo aparecen una vez
    """

    stopwords = []
    with open("stopwords.txt", "r", encoding="utf-8") as f:     # Lista de stopwords del español (algunas)
        stopwords = [line.strip() for line in f]

    keywords = []
    for keyword in keywords_list:
        keywords.extend(re.findall(r'\w+', keyword.lower()))     # muy importante el metodo extend para no guardas listas dentro de "words"

    words = []
    for title in title_list:
        for word in normalize_title(title, stopwords):
            if len(word) > 3:
                words.append(word)


    num_videos = len(title_list)                 
    counter = Counter(words) + Counter(keywords)            # Suma las frecuencias de cada uno de los contadores

    
    return {
        word: {
            "amount": count,
            "percentage": round((count / num_videos) * 100, 2) if count < num_videos else 100
        }
        for word, count in counter.most_common() if count > 1
    }




def normalize_title(title, stopwords):                 # A partir de un titulo, devuelve una lista de palabras normalizadas

    # Lowercase text
    title = title.lower()

    # Quitar tildes
    title = unicodedata.normalize('NFD', title)
    title = title.encode('ascii', 'ignore').decode('utf-8')

    # Transform into word array and remove words of 3 words or less
    words = re.findall(r'\b\w{4,}\b', title)

    return [w for w in words if w not in stopwords]

