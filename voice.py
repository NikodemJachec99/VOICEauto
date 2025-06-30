import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
from dotenv import load_dotenv
from openai import OpenAI # <--- NOWY IMPORT

# --- Konfiguracja Aplikacji ---
load_dotenv()
st.set_page_config(layout="wide")

# --- Konfiguracja Klientów API ---
ELEVENLABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

# Inicjalizacja klienta OpenAI
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    st.warning("Klucz OPENAI_API_KEY nie został znaleziony w .env. Generowanie promptu przez AI będzie niemożliwe.")
    openai_client = None


# --- Funkcje API ElevenLabs ---

@st.cache_data(ttl=3600) # Cache na 1 godzinę
def get_available_voices(api_key):
    """Pobiera listę dostępnych głosów z API ElevenLabs."""
    if not api_key:
        st.error("Klucz API ElevenLabs nie jest ustawiony. Sprawdź plik .env.")
        return {}
    try:
        # Używamy prostego requests, żeby uniknąć zależności od całej biblioteki elevenlabs
        response = requests.get(
            f"{ELEVENLABS_BASE_URL}/voices",
            headers={"xi-api-key": api_key}
        )
        response.raise_for_status()
        voices_data = response.json().get("voices", [])
        return {voice['name']: voice['voice_id'] for voice in voices_data}
    except Exception as e:
        st.error(f"Nie udało się pobrać głosów z ElevenLabs: {e}")
        return {}

def create_elevenlabs_agent(api_key, agent_name, system_prompt, voice_id, language, first_message):
    """Tworzy agenta konwersacyjnego i zwraca jego ID oraz link do widgetu."""
    if not api_key:
        st.error("Klucz API ElevenLabs nie jest ustawiony.")
        return None, None

    headers = {
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    # POPRAWIONY URL - dodano /create na końcu
    agent_creation_url = f"{ELEVENLABS_BASE_URL}/convai/agents/create"

    # Minimalna struktura danych z przykładu API
    # Oczyszczanie tekstów z problematycznych znaków
    clean_system_prompt = system_prompt.replace('\n\n', ' ').replace('\n', ' ').strip()
    clean_first_message = first_message.replace('\n\n', ' ').replace('\n', ' ').strip()
    
    # Logowanie danych wejściowych
    st.write("Debug - Input data:")
    st.write(f"system_prompt: {clean_system_prompt}")
    st.write(f"first_message: {clean_first_message}")
    st.write(f"language: {language}")
    st.write(f"voice_id: {voice_id}")
    
    data = {
        
        "name": agent_name,
        "conversation_config": {
            "tts": {
        "voice_id": voice_id},
                "agent": {
                "first_message": clean_first_message,
                "prompt": {
                    "prompt": clean_system_prompt,
                    "llm": "gpt-4o",  
                    "built_in_tools": {
            "language_detection": {
              "name": "language_detection",
              "description": "wykryj język",
              "params": {}
            },
                }
            },       
        },
        "language_presets": {
        "pl": {
          "overrides": {
            "agent": {
              
            }
          }
        }
      }
      
    }
    }


    
    try:
        # Metoda to POST
        response = requests.post(agent_creation_url, json=data, headers=headers)
        
        # Sprawdzanie statusu odpowiedzi
        response.raise_for_status()
        
        agent_data = response.json()
        agent_id = agent_data.get("agent_id")
        
        if not agent_id:
            st.error("Odpowiedź API nie zawierała agent_id.")
            st.json(agent_data) # Pokaż całą odpowiedź, jeśli brakuje ID
            return None, None
        
        # Tworzymy poprawny URL do testowania agenta
        public_widget_url = f"https://elevenlabs.io/app/talk-to?agent_id={agent_id}"
        
        return agent_id, public_widget_url
    
    except requests.exceptions.HTTPError as http_err:
        # Bardziej szczegółowe logowanie błędu
        st.error(f"Błąd HTTP podczas tworzenia agenta: {http_err}")
        st.error(f"URL: {http_err.request.url}")
        st.error(f"Metoda: {http_err.request.method}")
        st.error(f"Odpowiedź serwera: {http_err.response.text}")
        return None, None
    except Exception as e:
        st.error(f"Wystąpił nieoczekiwany błąd: {e}")
        return None, None

# --- MAPOWANIE JĘZYKÓW ---
LANGUAGE_MAPPING = {
    "Polski": "pl",
    "English": "en", 
    "Deutsch": "de",
    "Español": "es",
    "Français": "fr"
}

# --- FUNKCJE DO GENEROWANIA PROMPTU I FIRST MESSAGE PRZEZ GPT ---
def generate_system_prompt_with_gpt(role, tone, scraped_text, language):
    """Generuje systemowy prompt dla bota przy użyciu API OpenAI."""
    if not openai_client:
        st.error("Klient OpenAI nie jest skonfigurowany. Nie można wygenerować promptu.")
        return None

    # Ograniczamy tekst, żeby nie przekroczyć limitu tokenów
    truncated_text = scraped_text[:15000]

    meta_prompt = f"""
    Jesteś ekspertem w tworzeniu promptów systemowych dla voicebotów AI.
    Twoim zadaniem jest stworzenie zwięzłego i efektywnego promptu na podstawie poniższych wytycznych.
    
    Wytyczne dla voicebota:
    1.  Rola: {role}
    2.  Ton: {tone}
    3.  Język: {language}
    4.  Baza Wiedzy: Voicebot musi odpowiadać na pytania WYŁĄCZNIE na podstawie poniższego tekstu. Nie może wymyślać informacji. 
    5.  Brak wiedzy: Jeśli odpowiedź nie znajduje się w tekście, bot musi jasno i grzecznie poinformować, że nie posiada takich informacji.
    6.  Zwięzłość: Odpowiedzi powinny być krótkie i na temat nie mogą dotyczyć konkurencji.
    7.  Język odpowiedzi: Bot musi odpowiadać w języku {language}.
    8.  Długość: Prompt ma być jak najdłuższy mają to byś wszystkie produkty cała oferta firmy 

    Oto tekst bazy wiedzy:
    ---
    {truncated_text}
    ---

    Wygeneruj teraz prompt systemowy dla tego voicebota w języku {language}, który będzie jego wewnętrzną instrukcją. Zacznij od słów "Jesteś pomocnym asystentem...".
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": "Jesteś światowej klasy ekspertem od tworzenia promptów dla AI."},
                {"role": "user", "content": meta_prompt}
            ]
        )
        generated_prompt = response.choices[0].message.content
        return generated_prompt.strip()
    except Exception as e:
        st.error(f"Błąd podczas generowania promptu przez OpenAI: {e}")
        return None

def generate_first_message_with_gpt(role, tone, language, scraped_text):
    """Generuje pierwszą wiadomość bota przy użyciu API OpenAI."""
    if not openai_client:
        st.error("Klient OpenAI nie jest skonfigurowany. Nie można wygenerować pierwszej wiadomości.")
        return None

    # Ograniczamy tekst, żeby nie przekroczyć limitu tokenów
    truncated_text = scraped_text[:5000]  # Krótszy tekst dla first message

    meta_prompt = f"""
    Jesteś ekspertem w tworzeniu pierwszych wiadomości dla voicebotów AI.
    Twoim zadaniem jest stworzenie krótkiej, przyjaznej pierwszej wiadomości powitalnej.
    
    Wytyczne:
    1. Rola bota: {role}
    2. Ton: {tone}
    3. Język: {language}
    4. Długość: Maksymalnie 2-3 zdania
    5. Cel: Powitać użytkownika i krótko wyjaśnić, w czym bot może pomóc
    6. Bazuj na treści strony: {truncated_text[:1000]}...
    
    Wygeneruj krótką, przyjazną wiadomość powitalną w języku {language}, którą bot powie jako pierwszą rzecz do użytkownika.
    Nie używaj formatowania markdown ani znaków specjalnych - tylko czysty tekst.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Jesteś ekspertem od tworzenia przyjaznych wiadomości powitalnych dla botów."},
                {"role": "user", "content": meta_prompt}
            ]
        )
        generated_message = response.choices[0].message.content
        return generated_message.strip()
    except Exception as e:
        st.error(f"Błąd podczas generowania pierwszej wiadomości przez OpenAI: {e}")
        return None


# --- Funkcje do Scrapowania Stron  ---

def get_all_links(url, session, base_netloc):
    try:
        response = session.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        links = set()
        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']
            joined_link = urljoin(url, link)
            if urlparse(joined_link).netloc == base_netloc and joined_link.startswith('http'):
                links.add(joined_link)
        return links
    except requests.exceptions.RequestException:
        return set()

def scrape_text(url, session):
    try:
        response = session.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        return " ".join(soup.stripped_strings)
    except requests.exceptions.RequestException:
        return ""

def crawl_website(start_url, max_pages):
    to_visit = {start_url}
    visited = set()
    all_text = ""
    base_netloc = urlparse(start_url).netloc
    progress_bar = st.progress(0, text=f"Przygotowuję do scrapowania... limit: {max_pages} stron")
    status_text = st.empty()
    with requests.Session() as session:
        while to_visit and len(visited) < max_pages:
            current_url = to_visit.pop()
            if current_url not in visited:
                visited.add(current_url)
                status_text.text(f"Scrapuję stronę {len(visited)}/{max_pages}: {current_url}")
                text = scrape_text(current_url, session)
                if text:
                    all_text += text + "\n\n"
                if len(visited) < max_pages:
                    new_links = get_all_links(current_url, session, base_netloc)
                    to_visit.update(new_links - visited)
                progress_bar.progress(len(visited) / max_pages, text=f"Postęp: {len(visited)}/{max_pages} stron")

    status_text.text(f"Scraping zakończony! Odwiedzono {len(visited)} stron.")
    progress_bar.progress(1.0, text="Ukończono!")
    return all_text

# --- Główna Aplikacja Streamlit ---

st.title("Generator Voicebotów ElevenLabs ze Stron Internetowych")
st.markdown("Ta aplikacja automatycznie tworzy zaawansowanego agenta konwersacyjnego, wykorzystując treść Twojej strony internetowej oraz AI do generowania jego osobowości.")

available_voices = get_available_voices(ELEVENLABS_API_KEY)

if not available_voices:
    st.error("Nie udało się załadować głosów z ElevenLabs. Aplikacja nie może kontynuować.")
    st.stop()

with st.form("voicebot_form"):
    st.subheader("Krok 1: Skonfiguruj Agenta i Scrapowanie")
    col1, col2 = st.columns(2)
    with col1:
        website_url = st.text_input("Wprowadź URL strony do scrapowania:", "https://elevenlabs.io/docs/conversational-ai/overview")
    with col2:
        max_pages_to_scrape = st.slider("Maksymalna liczba podstron do scrapowania:", min_value=1, max_value=100, value=10, help="Crawler zatrzyma się po zescrapowaniu tej liczby unikalnych podstron.")
    
    agent_name = st.text_input("Nazwij swojego agenta:", "Asystent Strony WWW")

    st.subheader("Krok 2: Zdefiniuj Osobowość Bota")
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        persona_tone = st.selectbox("Wybierz ton głosu bota:", ["Formalny", "Niefolmalny/Luźny", "Profesjonalny", "Przyjacielski"])
    with col4:
        persona_role = st.selectbox("Wybierz rolę bota:", options=["Doradca klienta", "Sprzedawca", "Asystent Q&A", "Przewodnik po stronie"])
    with col5:
        selected_language = st.selectbox("Wybierz język bota:", ["Polski", "English", "Deutsch", "Español", "Français"])
    with col6:
        selected_voice_name = st.selectbox("Wybierz głos dla bota:", list(available_voices.keys()))
        selected_voice_id = available_voices[selected_voice_name]

    submitted = st.form_submit_button("Uruchom Magię ✨ (Utwórz Voicebota)", type="primary", use_container_width=True)

if submitted:
    if not website_url:
        st.warning("Proszę wprowadzić URL strony internetowej.")
    elif not openai_client:
         st.error("Nie można kontynuować bez skonfigurowanego klienta OpenAI. Sprawdź swój klucz API.")
    else:
        scraped_text = crawl_website(website_url, max_pages_to_scrape)

        if scraped_text:
            st.success("Strona została pomyślnie zescrapowana!")
            
            with st.spinner("Sztuczna inteligencja (GPT) tworzy teraz idealny prompt i pierwszą wiadomość dla Twojego bota..."):
                system_prompt = generate_system_prompt_with_gpt(persona_role, persona_tone, scraped_text, selected_language)
                first_message = generate_first_message_with_gpt(persona_role, persona_tone, selected_language, scraped_text)
            
            if system_prompt:
                st.success("Prompt systemowy został wygenerowany przez AI!")
                
                # Sprawdź czy first_message został wygenerowany
                if first_message:
                    st.success("Pierwsza wiadomość została wygenerowana przez AI!")
                else:
                    st.warning("Nie udało się wygenerować pierwszej wiadomości, ale kontynuujemy z samym promptem.")
                    first_message = ""  # Ustaw pustą wartość
                
                with st.expander("Zobacz prompt i pierwszą wiadomość wygenerowane przez AI"):
                    st.text_area("System Prompt", system_prompt, height=200)
                    if first_message:
                        st.text_area("Pierwsza wiadomość", first_message, height=100)

                with st.spinner("Tworzenie agenta w ElevenLabs i generowanie linku..."):
                    # Konwertuj język na kod ISO
                    language_code = LANGUAGE_MAPPING.get(selected_language, "en")
                    
                    agent_id, public_widget_url = create_elevenlabs_agent(
                        ELEVENLABS_API_KEY,
                        agent_name,
                        system_prompt,
                        selected_voice_id,
                        language_code,
                        first_message
                    )

                if agent_id and public_widget_url:
                    st.success("Voicebot został pomyślnie utworzony!")
                    st.balloons()
                    st.subheader("Gotowe! Oto Twój Agent:")
                    st.markdown(f"**ID Agenta:** `{agent_id}`")
                    st.markdown(f"**Publiczny link do widgetu testowego:**")
                    st.link_button("Testuj Voicebota", public_widget_url)
                    st.info("Kliknij w powyższy przycisk, aby otworzyć i przetestować swojego nowego voicebota.")
                else:
                    st.error("Nie udało się utworzyć voicebota. Sprawdź komunikaty błędów powyżej.")
            else:
                 st.error("Nie udało się wygenerować promptu przez AI. Sprawdź błędy powyżej.")
        else:
            st.error("Nie udało się pobrać żadnej treści ze strony. Sprawdź URL i spróbuj ponownie.")