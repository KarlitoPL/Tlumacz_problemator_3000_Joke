import streamlit as st
import requests
import uuid
import time
import bcrypt
from dotenv import dotenv_values
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from openai import OpenAI
import openai

# -------------------------  KONFIGURACJA  -------------------------

env = dotenv_values(".env")

# Nadpisujemy wartości zmiennych ze st.secrets, jeśli są dostępne
if "QDRANT_URL" in st.secrets:
    env["QDRANT_URL"] = st.secrets["QDRANT_URL"]
if "QDRANT_API_KEY" in st.secrets:
    env["QDRANT_API_KEY"] = st.secrets["QDRANT_API_KEY"]
if "OPENAI_API_KEY" in st.secrets:
    env["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Teraz możesz bezpiecznie zainicjować klienta OpenAI
openai_client = OpenAI(api_key=env["OPENAI_API_KEY"])

@st.cache_resource
def get_qdrant_client():
    """
    Zwraca zcache'owaną instancję klienta Qdrant.
    """
    return QdrantClient(
        url=env["QDRANT_URL"],     # np. "https://xxx-xxxxx-xxx-xxxxx.aws.cloud.qdrant.io"
        api_key=env["QDRANT_API_KEY"],
    )


# -------------------------  FUNKCJE: UŻYTKOWNICY  -------------------------

def create_users_collection():
    """
    Tworzy (jeśli nie istnieje) kolekcję 'users' do przechowywania danych o użytkownikach.
    """
    client = get_qdrant_client()
    existing = client.get_collections().collections
    names = [col.name for col in existing]
    if "users" not in names:
        client.create_collection(
            collection_name="users",
            vectors_config=VectorParams(size=1, distance=Distance.COSINE)
        )

def hash_password(password: str) -> str:
    """
    Hashuje hasło przy użyciu bcrypt.
    Zwraca zakodowany hash w formie str.
    """
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")

def check_password(password: str, hashed: str) -> bool:
    """
    Sprawdza, czy hasło 'password' pasuje do zapisanego hasha 'hashed'.
    """
    return bcrypt.hashpw(password.encode("utf-8"), hashed.encode("utf-8")) == hashed.encode("utf-8")

def register_user(username: str, password: str, email: str) -> bool:
    """
    Rejestruje nowego użytkownika w kolekcji 'users'.
    Zwraca True, jeśli rejestracja się udała, lub False, jeśli użytkownik już istnieje.
    """
    client = get_qdrant_client()
    if find_user(username):
        return False  # użytkownik już istnieje

    user_id = str(uuid.uuid4())
    payload = {
        "username": username,
        "hashed_password": hash_password(password),
        "email": email,
        "created_at": int(time.time())
    }
    client.upsert(
        collection_name="users",
        points=[{
            "id": user_id,
            "vector": [0.0],  # Dummy wektor
            "payload": payload
        }]
    )
    return True

def find_user(username: str):
    client = get_qdrant_client()

    all_points = []
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name="users",
            limit=50,
            offset=next_offset
        )
        all_points.extend(points)

        # Jeśli next_offset jest None, nie ma kolejnych "stron"
        if next_offset is None:
            break

    for point in all_points:
        payload = point.payload
        if payload.get("username") == username:
            return payload
    return None

# Dodane funkcje do resetu hasła:

def find_user_record(username: str):
    """
    Zwraca CAŁY rekord (point) użytkownika o danej nazwie (z polami .id i .payload).
    Jeśli nie znajdzie, zwraca None.
    """
    client = get_qdrant_client()
    all_points = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name="users",
            limit=50,
            offset=next_offset
        )
        all_points.extend(points)
        if next_offset is None:
            break
    for point in all_points:
        if point.payload.get("username") == username:
            return point
    return None

def reset_password(username: str, email: str, new_password: str) -> bool:
    """
    Resetuje hasło użytkownika, jeśli podany adres e-mail zgadza się z tym zapisanym w bazie.
    Aktualizuje zahashowane hasło w Qdrant i zwraca True, jeśli operacja się powiodła.
    """
    client = get_qdrant_client()
    user_record = find_user_record(username)
    if not user_record:
        return False
    if user_record.payload.get("email") != email:
        return False

    new_hashed = hash_password(new_password)
    updated_payload = user_record.payload.copy()
    updated_payload["hashed_password"] = new_hashed

    client.upsert(
         collection_name="users",
         points=[{
              "id": user_record.id,
              "vector": [0.0],
              "payload": updated_payload
         }]
    )
    return True

def login_user(username: str, password: str) -> bool:
    """
    Loguje użytkownika. Zwraca True, jeśli dane poprawne, False w przeciwnym wypadku.
    """
    user = find_user(username)
    if not user:
        return False
    hashed = user["hashed_password"]
    return check_password(password, hashed)

def create_user_collection_if_not_exists(username: str):
    """
    Dla zalogowanego użytkownika tworzy kolekcję w Qdrant,
    w której będą przechowywane jego historie.
    """
    client = get_qdrant_client()
    existing = client.get_collections().collections
    collection_names = [col.name for col in existing]
    if username not in collection_names:
        client.create_collection(
            collection_name=username,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE)
        )

# -------------------------  FUNKCJE: GENERATOR HISTORII  -------------------------

def save_history(account_name, record_type, prompt_text, generated_text, cost_usd, cost_pln):
    """
    Zapisuje rekord (historię) do kolekcji odpowiadającej kontu w Qdrant.
    """
    client = get_qdrant_client()
    record_id = str(uuid.uuid4())
    vector = [0.0]
    payload = {
        "type": record_type,
        "prompt": prompt_text,
        "generated_text": generated_text,
        "cost_usd": cost_usd,
        "cost_pln": cost_pln,
        "timestamp": int(time.time())
    }
    client.upsert(
        collection_name=account_name,
        points=[{
            "id": record_id,
            "vector": vector,
            "payload": payload
        }]
    )
    st.success(f"Twoja opowieść została pomyślnie zapisana na Twoim koncie!")

def get_history(account_name):
    client = get_qdrant_client()
    all_points = []
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=account_name,
            limit=50,
            offset=next_offset
        )
        all_points.extend(points)

        if next_offset is None:
            break

    return all_points

def generate_prompt(problem, mode, style=None):
    """
    Generuje prompt na podstawie wpisanego problemu oraz wybranego trybu.
    """
    if mode == "Ulubione Universum":
        prompt = ("Stwórz mi z tego problemu szczegółowe opowiadanie dla osoby, która go nie rozumie, aby mu wytłumaczyć.")
        if style:
            prompt += f" Ale w stylu opowiadań {style}."
    elif mode == "Zabawnie":
        prompt = ("Stwórz mi z tego problemu długi, kilkupoziomowy żart, ale taki śmieszny jak w try not laugh, "
                  "dla osoby, która go nie rozumie, aby mu szczegółowo i po ludzku wytłumaczyć jak, bez żadnych innych wstawek i wytłumaczeń.")
    else:
        prompt = ""
    prompt += "\nProblem: " + problem
    return prompt

def get_gpt4_response(prompt):
    """
    Wywołuje API GPT-4 (lub inny model), aby wygenerować odpowiedź na zadany prompt.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # Zmień na "gpt-4" lub inny model, do którego masz dostęp
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Błąd przy generowaniu odpowiedzi: {e}")
        return None

def approximate_token_count(text):
    """
    Prosta aproksymacja liczby tokenów: średnio 4 znaki = 1 token.
    """
    return max(1, len(text) // 4)

def calculate_cost(prompt_text, generated_text, exchange_rate=4):
    """
    Oblicza przybliżony koszt wywołania API GPT-4:
      - 2.5 USD za milion tokenów dla prompta,
      - 10.0 USD za milion tokenów dla wygenerowanego tekstu.
    Zwraca koszt w USD oraz przeliczony na złote.
    """
    prompt_tokens = approximate_token_count(prompt_text)
    output_tokens = approximate_token_count(generated_text)
    cost_prompt_usd = prompt_tokens * 2.5 / 1_000_000
    cost_output_usd = output_tokens * 10.0 / 1_000_000
    total_cost_usd = cost_prompt_usd + cost_output_usd
    total_cost_pln = total_cost_usd * exchange_rate
    return total_cost_usd, total_cost_pln


def simulate_loading():
    placeholder = st.empty()
    messages = [
        "Problem wysłany do mistrza pióra...",
        "Mistrz Pióra główkuje...",
        "Ok, to interesujące...",
        "Hmm... niezłe...",
        "Gotowe, już do ciebie leci!"
    ]
    for msg in messages:
        placeholder.text(msg)
        time.sleep(2)  # pauza 2 sekundy między komunikatami
    # Po zakończeniu można wyczyścić placeholder:
    placeholder.empty()


# -------------------------  INTERFEJS STREAMLIT  -------------------------

def main():
    st.title("Mistrz Pióra: Rozwiąże Twoje Problemy w Rozrywkowy Sposób")

    # Upewniamy się, że istnieje kolekcja 'users'
    create_users_collection()

    # Inicjalizacja zmiennych sesyjnych
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = ""

    # Cztery zakładki: Logowanie/Rejestracja, Reset Hasła, Generowanie, Historia
    tab_auth, tab_reset, tab_generate, tab_saved = st.tabs([
        "Logowanie / Rejestracja", "Reset Hasła", "Generuj", "Zapisane Opowieści i Żarty"
    ])

    # -------------------------  Zakładka 1: Logowanie / Rejestracja  -------------------------
    with tab_auth:
        st.subheader("Zarejestruj się lub zaloguj")
        auth_mode = st.radio("Wybierz akcję:", ["Zarejestruj", "Zaloguj"])
        username = st.text_input("Nazwa użytkownika:", key="auth_username")
        password = st.text_input("Hasło:", type="password", key="auth_password")
        email = None
        if auth_mode == "Zarejestruj":
            email = st.text_input("Adres e‑mail:", key="auth_email")

        if st.button("Dalej", key="auth_button"):
            if not username or not password or (auth_mode == "Zarejestruj" and not email):
                st.error("Podaj wszystkie wymagane dane!")
            else:
                if auth_mode == "Zarejestruj":
                    success = register_user(username, password, email)
                    if success:
                        st.success("Rejestracja udana! Możesz się teraz zalogować.")
                    else:
                        st.warning("Użytkownik o tej nazwie już istnieje.")
                else:
                    if login_user(username, password):
                        st.success("Zalogowano pomyślnie!")
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = username
                        create_user_collection_if_not_exists(username)
                    else:
                        st.error("Błędna nazwa użytkownika lub hasło. Jeśli nie pamiętasz hasła, przejdź do zakładki Reset Hasła")

    # -------------------------  Zakładka 2: Reset Hasła  -------------------------
    with tab_reset:
        st.subheader("Reset Hasła")
        reset_username = st.text_input("Nazwa użytkownika:", key="reset_username")
        reset_email = st.text_input("Adres e‑mail:", key="reset_email")
        new_password = st.text_input("Nowe hasło:", type="password", key="reset_new_password")
        confirm_password = st.text_input("Potwierdź nowe hasło:", type="password", key="reset_confirm_password")
        if st.button("Resetuj hasło", key="reset_button"):
            if not reset_username or not reset_email or not new_password or not confirm_password:
                st.error("Wypełnij wszystkie pola!")
            elif new_password != confirm_password:
                st.error("Nowe hasło i potwierdzenie nie są zgodne!")
            else:
                if reset_password(reset_username, reset_email, new_password):
                    st.success("Hasło zostało zresetowane. Możesz się teraz zalogować.")
                else:
                    st.error("Reset hasła nie powiódł się. Sprawdź dane.")

    # -------------------------  Zakładka 3: Generowanie  -------------------------
    with tab_generate:
        st.header("Generowanie opowieści lub żartów")
        if not st.session_state["logged_in"]:
            st.warning("Musisz się zalogować, aby korzystać z tej zakładki.")
        else:
            st.info(f"Zalogowano jako: {st.session_state['username']}")
            problem_input = st.text_area("Wpisz problem lub cokolwiek, czego nie rozumiesz:")
            mode = st.radio("Wybierz tryb:", ("Ulubione Universum", "Zabawnie"))
            style_input = None
            if mode == "Ulubione Universum":
                style_input = st.text_input("Wpisz ulubionego pisarza, reżysera lub tytuł ulubionego filmu:")

            if st.button("Generuj", key="generate_button"):
                if not problem_input:
                    st.error("Pole problem jest puste!")
                else:
                    prompt = generate_prompt(problem_input, mode, style_input)

                    # Wyświetlenie komunikatów symulujących pracę "mistrza pióra"
                    simulate_loading()

                    generated_text = get_gpt4_response(prompt)
                    if generated_text:
                        st.subheader("Oto Twoja unikalna opowieść:")
                        st.write(generated_text)
                        
                        cost_usd, cost_pln = calculate_cost(prompt, generated_text)
                        st.markdown("---")
                        st.write(f"**Szacowany koszt to około  (~{cost_pln:.2f} zł)")
                        save_history(
                            account_name=st.session_state["username"],
                            record_type=mode,
                            prompt_text=prompt,
                            generated_text=generated_text,
                            cost_usd=cost_usd,
                            cost_pln=cost_pln
                        )

    # -------------------------  Zakładka 4: Historia  -------------------------
    with tab_saved:
        st.header("Zapisane Opowieści i Żarty")
        if not st.session_state["logged_in"]:
            st.warning("Musisz się zalogować, aby przeglądać zapisane historie.")
        else:
            st.info(f"Zalogowano jako: {st.session_state['username']}")
            user_collection = st.session_state["username"]
            client = get_qdrant_client()
            existing = client.get_collections().collections
            collection_names = [col.name for col in existing]
            if user_collection in collection_names:
                history = get_history(user_collection)
                if history:
                    for item in history:
                        payload = item.payload
                        st.write("---")
                        st.write(f"**Typ**: {payload.get('type')}")
                        st.write(f"**Tekst**: {payload.get('generated_text')}")
                        cost_usd = payload.get("cost_usd", 0)
                        cost_pln = payload.get("cost_pln", 0)
                        st.write(f"**Szacowany koszt to około  (~{cost_pln:.2f} zł)")
                else:
                    st.info("Brak zapisanych rekordów w Twojej kolekcji.")
            else:
                st.warning("Twoja kolekcja jeszcze nie istnieje lub nie została poprawnie utworzona.")

if __name__ == "__main__":
    main()
