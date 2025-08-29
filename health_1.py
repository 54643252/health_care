import streamlit as st
from snowflake.snowpark import Session
from cryptography.hazmat.primitives import serialization
# ------------------------
# PAGE CONFIG
# ------------------------
# ------------------------
# SNOWFLAKE CONNECTION (cached)
# ------------------------
@st.cache_resource
def get_connection():
    private_key_bytes = st.secrets["snowflake"]["private_key"].encode()
    private_key = serialization.load_pem_private_key(private_key_bytes, password=None)
    return snowflake.connector.connect(
        account=st.secrets["snowflake"]["account"],
        user=st.secrets["snowflake"]["user"],
        role=st.secrets["snowflake"]["role"],
        warehouse=st.secrets["snowflake"]["warehouse"],
        database=st.secrets["snowflake"]["database"],
        schema=st.secrets["snowflake"]["schema"],
        private_key=private_key,
    )
 
conn = get_connection()
cursor = conn.cursor()
st.set_page_config(page_title="Disease Progression Model", layout="wide")
st.markdown("""
<style>
    /* Sidebar container full height with vertical space distribution */
    section[data-testid="stSidebar"] > div:first-child {
        height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    /* Logo area at the top, centered */
    .sidebar-top {
        flex: 0 0 auto;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px 0 10px 0;
    }

    /* About Us area in the middle, centered text */
    .sidebar-middle {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 20px;
        text-align: center;
    }
    .sidebar-middle h5 {
        margin-bottom: 12px;
    }
    .sidebar-middle p {
        margin: 0 auto;
        font-size: 1rem;
        line-height: 1.4;
        max-width: 270px;
    }

    /* Social icons area at the bottom, spaced and centered */
    .sidebar-bottom {
        flex: 0 0 auto;
        display: flex;
        justify-content: center;
        gap: 18px;
        padding: 24px 0;
    }
    .sidebar-bottom img {
        vertical-align: middle;
    }

    /* Main heading styling, centered */
    .main-heading {
        font-size: 2.8rem;
        font-weight: 700;
        margin-top: 40px;
        margin-bottom: 30px;
        text-align: center;
        letter-spacing: 0.8px;
    }

    /* Chat message tweaks to hide avatar and align text flush left */
    div[data-testid="stChatMessage"] > div:nth-child(1) {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        visibility: hidden !important;
    }
    div[data-testid="stChatMessage"] > div:nth-child(2) {
        margin-left: 0 !important;
        padding-left: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.markdown("""
<div class="sidebar-top">
    <img src="https://booleandata.com/wp-content/uploads/2022/09/Boolean-logo_Boolean-logo-USA-1-980x316.png" style="max-width:90%;">
</div>
<div class="sidebar-middle">
    <div>
        <h5>🚀 About Us</h5>
        <p>We are a data-driven company revolutionizing the insurance industry through predictive analytics.<br>
        Our models help detect fraudulent claims with high accuracy and transparency.</p>
    </div>
</div>
<div class="sidebar-bottom">
    <a href="https://booleandata.ai/" target="_blank" title="Website">🌐</a>
    <a href="https://www.facebook.com/Booleandata" target="_blank" title="Facebook">
        <img src="https://cdn-icons-png.flaticon.com/24/1384/1384005.png" width="24" alt="Facebook">
    </a>
    <a href="https://www.youtube.com/channel/UCd4PC27NqQL5v9-1jvwKE2w" target="_blank" title="YouTube">
        <img src="https://cdn-icons-png.flaticon.com/24/1384/1384060.png" width="24" alt="YouTube">
    </a>
    <a href="https://www.linkedin.com/company/boolean-data-systems" target="_blank" title="LinkedIn">
        <img src="https://cdn-icons-png.flaticon.com/24/145/145807.png" width="24" alt="LinkedIn">
    </a>
</div>
""", unsafe_allow_html=True)
# ------------------------


# ------------------------
# SNOWFLAKE CONNECTION
# ------------------------
connection_parameters = {
    "account": "YOUR_ACCOUNT",
    "user": "YOUR_USER",
    "password": "YOUR_PASSWORD",
    "role": "ACCOUNTADMIN",
    "warehouse": "COMPUTE_WH",
    "database": "HEALTH_CARE",
    "schema": "HEALTH_CARE_SCHEMA"
}

session = Session.builder.configs(connection_parameters).create()

# ------------------------
# STATE MANAGEMENT
# ------------------------
if "chats" not in st.session_state:
    st.session_state.chats = []
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

st.title("Disease Progression Assistant")

if st.session_state.current_chat is None:
    current_messages = []
else:
    current_messages = st.session_state.chats[st.session_state.current_chat]["messages"]

# Render past chat
for role, message in current_messages:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(message)

# ------------------------
# CHAT INPUT
# ------------------------
user_input = st.chat_input("Ask about patient progression, treatments, or prognosis...")

if user_input:
    if st.session_state.current_chat is None:
        chat_name = f"{user_input[:30]}..."
        st.session_state.chats.append({"name": chat_name, "messages": []})
        st.session_state.current_chat = len(st.session_state.chats) - 1

    st.session_state.chats[st.session_state.current_chat]["messages"].append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build conversation history
    history_messages = st.session_state.chats[st.session_state.current_chat]["messages"]
    conversation_history = "\n".join(
        [f"{role.capitalize()}: {msg}" for role, msg in history_messages if role != "assistant"]
    )

    # RAG query
    query = f"""
    WITH query AS (
      SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(
        'snowflake-arctic-embed-m',
        '{user_input}'
      ) AS q_vec
    ),
    retrieved AS (
      SELECT
        patient_id,
        disease_type,
        disease_stage,
        visit_date,
        clinical_summary,
        assigned_treatment,
        prognosis_summary
      FROM health_care.health_care_schema.health_care_table, query
      ORDER BY VECTOR_COSINE_SIMILARITY(health_care_vector, q_vec) DESC
      LIMIT 5
    ),
    context AS (
      SELECT LISTAGG(
        CONCAT(
          'Patient ID: ', patient_id, 
          ', Disease: ', disease_type, ' ', disease_stage, 
          ', Visit: ', visit_date, 
          ', Summary: ', clinical_summary, 
          ', Treatment: ', assigned_treatment, 
          ', Prognosis: ', prognosis_summary
        ), '\\n') AS ctx
      FROM retrieved
    )
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
        'claude-3-7-sonnet',
        CONCAT(
          'You are a medical expert. Use the conversation history and retrieved patient records to answer clearly.\\n\\n',
          'Conversation history:\\n',
          '{conversation_history}',
          '\\n\\nContext:\\n',
          ctx,
          '\\n\\nLatest Question:\\n',
          '{user_input}'
        )
    ) AS answer
    FROM context;
    """

    result = session.sql(query).collect()
    answer = result[0]["ANSWER"]

    st.session_state.chats[st.session_state.current_chat]["messages"].append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.rerun()

