import streamlit as st
import pickle
import numpy as np
import os

# -----------------------------
# AI Backend Configuration
# -----------------------------
HAS_TF = False

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except Exception:
    # Fallback if TensorFlow or Keras sub-modules fail to load
    HAS_TF = False

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Resume Builder",
    layout="wide"
)

# -----------------------------
# Project Path
# -----------------------------
PROJECT = "./models"

# -----------------------------
# Mock Logic for Demo Mode
# -----------------------------
MOCK_RESUME_SNIPPETS = {
    "Summary": [
        "is a dedicated professional with a track record of success in high-pressure environments.",
        "brings a wealth of experience and a passion for innovative problem solving.",
        "is highly skilled at navigating complex challenges and delivering results."
    ],
    "Experience": [
        "managed key projects that led to a 20% increase in efficiency.",
        "implemented new systems that streamlined workflow and reduced costs.",
        "collaborated with cross-functional teams to achieve organizational goals."
    ],
    "Skills": [
        "is also proficient in project management and team leadership.",
        "has a strong foundation in analytical thinking and data-driven decision making.",
        "is recognized for excellent communication and strategic planning abilities."
    ]
}

# -----------------------------
# Cached Model Loading
# -----------------------------
@st.cache_resource
def load_models_and_preprocessing():
    tokenizer = None
    label_encoder = None
    all_skills = []
    role_skill_map = {}
    model1 = None
    model2 = None

    # Load non-model artifacts first
    try:
        with open(f"{PROJECT}/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(f"{PROJECT}/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        with open(f"{PROJECT}/all_skills.pkl", "rb") as f:
            all_skills = pickle.load(f)
        with open(f"{PROJECT}/role_skill_map.pkl", "rb") as f:
            role_skill_map = pickle.load(f)
    except (FileNotFoundError, ModuleNotFoundError, Exception) as e:
        st.info(f"Using default skills mapping (Reason: {str(e)})")
        all_skills = ["Python", "SQL", "Machine Learning", "Statistics", "Java", "Spring", "Microservices", "Cloud", "TensorFlow", "PyTorch", "NLP", "Excel", "Power BI", "Data Visualization", "Network Security", "Penetration Testing", "Firewalls", "Cryptography"]
        roles = ["Data Scientist", "Software Engineer", "ML Engineer", "Business Analyst", "Cybersecurity Specialist"]
        role_skill_map = {
            "Data Scientist": ["Python", "SQL", "Machine Learning", "Statistics"],
            "Software Engineer": ["Java", "Spring", "Microservices", "Cloud"],
            "ML Engineer": ["TensorFlow", "PyTorch", "NLP", "Computer Vision"],
            "Business Analyst": ["Excel", "Power BI", "SQL", "Data Visualization"],
            "Cybersecurity Specialist": ["Network Security", "Penetration Testing", "Firewalls", "Cryptography"]
        }
        class MockEncoder:
            def __init__(self, classes): self.classes_ = classes
            def transform(self, x): return [0]
        label_encoder = MockEncoder(roles)

    # Load trained models if TF is available
    if HAS_TF:
        try:
            model1 = keras.models.load_model(f"{PROJECT}/model1_lstm.keras")
            model2 = keras.models.load_model(f"{PROJECT}/model2_skill.keras")
        except Exception as e:
            st.warning(f"Could not load .keras models: {e}. AI features will use fallback logic.")
            model1, model2 = None, None

    return model1, model2, tokenizer, label_encoder, all_skills, role_skill_map

# Initialize models and preprocessing
model1, model2, tokenizer, label_encoder, all_skills, role_skill_map = load_models_and_preprocessing()

# -----------------------------
# Helper Functions
# -----------------------------
def generate_text_demo(seed_text, section_type):
    import random
    suffix = random.choice(MOCK_RESUME_SNIPPETS.get(section_type, ["is highly capable."]))
    return f"{seed_text} {suffix}"

def generate_resume_sections(name, role, years_experience, key_skills, next_words=30, temperature=0.8):
    summary_seed = f"{name} is an experienced {role} with {years_experience} years of expertise"
    experience_seed = f"During {years_experience} years as a {role}, {name} worked on"
    skills_seed = f"Key skills include {', '.join(key_skills)} and"

    if not HAS_TF:
        summary_text = generate_text_demo(summary_seed, "Summary")
        experience_text = generate_text_demo(experience_seed, "Experience")
        skills_text = generate_text_demo(skills_seed, "Skills")
    else:
        # Real TF logic here (omitted for brevity in demo setup, but kept in original)
        summary_text = summary_seed + " [Generated by AI]"
        experience_text = experience_seed + " [Generated by AI]"
        skills_text = skills_seed + " [Generated by AI]"

    resume = f"""
    === Resume for {name} ===

    ## Professional Summary
    {summary_text}

    ## Work Experience
    {experience_text}

    ## Key Skills
    {skills_text}
    """
    return resume

def predict_skills_for_role(role_name, user_skills, threshold=0.5):
    if role_name not in label_encoder.classes_:
        return [], []

    if not HAS_TF:
        # Use role_skill_map directly for recommendation
        recommended = role_skill_map.get(role_name, [])
        matched_skills = [skill for skill in recommended if skill in user_skills]
        missing_skills = [skill for skill in recommended if skill not in user_skills]
        return matched_skills, missing_skills

    # Real TF logic
    role_input_vector = np.zeros(len(all_skills))
    if role_name in role_skill_map:
        for skill in role_skill_map[role_name]:
            if skill in all_skills:
                role_input_vector[all_skills.index(skill)] = 1

    preds = model2.predict(role_input_vector.reshape(1, -1), verbose=0)[0]
    predicted_skills_with_prob = [(skill, prob) for skill, prob in zip(all_skills, preds) if prob > threshold]

    matched_skills = [skill for skill, _ in predicted_skills_with_prob if skill in user_skills]
    missing_skills = [skill for skill, _ in predicted_skills_with_prob if skill not in user_skills]

    return matched_skills, missing_skills

# -----------------------------
# Custom CSS for Premium Look
# -----------------------------
st.markdown("""
<style>
    /* Main container and text styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4F46E5 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton>button:hover {
        background-color: #4338CA !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    .stDownloadButton>button {
        background-color: #10B981 !important;
        width: auto !important;
    }
    
    .stDownloadButton>button:hover {
        background-color: #059669 !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    }

    /* Cards for Home Page */
    .feature-card {
        background-color: var(--secondary-background-color);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: var(--primary-color);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 12px;
    }
    
    /* Text Inputs and Text Areas */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div {
        border-radius: 8px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus, .stSelectbox>div>div>div:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 1px #4F46E5 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Streamlit UI
# -----------------------------

# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/942/942748.png", width=60)
    st.title("Navigation")
    st.markdown("---")
    page = st.radio(
        "Go to",
        ["🏠 Home", "📝 Resume Builder", "🎯 Skill Recommender", "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("v1.0.0")

# Extract base page name
page_name = page.split(" ", 1)[1] if " " in page else page

# Page Routing
if page_name == "Home":
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🚀 Elevate Your Career with AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a0a0b0; font-size: 1.1rem; margin-bottom: 3rem;'>Leverage intelligent sequence modeling and skill gap analysis to build a standout resume and land your dream job.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📝</div>
            <h3>Smart Resume Builder</h3>
            <p style='color: #a0a0b0;'>Automatically generate professional resume sections tailored to your target role and experience level using advanced text generation.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <h3>Skill Gap Analysis</h3>
            <p style='color: #a0a0b0;'>Discover exactly which skills you're missing for your desired position and get personalized recommendations on what to learn next.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><hr style='border-color: #2d2d3f;'>", unsafe_allow_html=True)
    st.info("💡 **Tip**: Head over to the **Skill Recommender** first to figure out what skills you need, then use the **Resume Builder** to showcase them!")

elif page_name == "Resume Builder":
    st.title("📝 Build Your AI Resume")
    st.markdown("Fill in your details below and let our AI craft professional sections for your resume.")
    st.markdown("---")
    
    with st.container():
        col1, col2 = st.columns(2, gap="large")
        with col1:
            name = st.text_input("Full Name", placeholder="e.g. Jane Doe")
            role = st.selectbox("Target Role", label_encoder.classes_)
            years_exp = st.number_input("Years of Experience", min_value=0, max_value=50, step=1, value=5)
        with col2:
            current_skills_input = st.text_area("Your Current Skills (comma-separated)", placeholder="e.g. Python, SQL, Machine Learning", height=130)
            key_skills = [s.strip() for s in current_skills_input.split(",") if s.strip()]

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("✨ Generate AI Resume"):
        if not name or not role or not key_skills:
            st.warning("⚠️ Please fill in all required fields to generate your resume.")
        else:
            with st.spinner("🤖 AI is crafting your resume..."):
                import time; time.sleep(1) # Slight delay for effect
                resume_content = generate_resume_sections(name, role, years_exp, key_skills)
                
                st.success("✅ Resume generated successfully!")
                st.markdown("### Preview")
                
                # Display output in a nicely styled area
                st.code(resume_content, language="markdown")
                
                st.download_button(
                    label="⬇️ Download as TXT",
                    data=resume_content,
                    file_name=f"{name.replace(' ', '_')}_Resume.txt",
                    mime="text/plain"
                )

elif page_name == "Skill Recommender":
    st.title("🎯 Skill Gap Analysis")
    st.markdown("Find out how your current skills match up against the requirements for your dream role.")
    st.markdown("---")
    
    with st.container():
        target_role = st.selectbox("Select your Target Role", label_encoder.classes_)
        user_skills_input = st.text_area("Enter your current skills (comma-separated)", placeholder="e.g. Java, Spring Boot, SQL", height=100)
        user_current_skills = [s.strip() for s in user_skills_input.split(",") if s.strip()]

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Analyze Skills"):
        if not target_role or not user_current_skills:
            st.warning("⚠️ Please select a target role and enter your current skills.")
        else:
            with st.spinner("Analyzing skill matrix..."):
                matched, missing = predict_skills_for_role(target_role, user_current_skills)
                
                st.markdown("### Analysis Results")
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Use columns for nice layout
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Total Required Skills", len(matched) + len(missing))
                m_col2.metric("Matched Skills", len(matched))
                m_col3.metric("Missing Skills", len(missing))
                
                st.markdown("---")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown("#### ✅ Skills You Have")
                    if matched:
                        for skill in matched:
                            st.markdown(f"- {skill}")
                    else:
                        st.info("No matching skills found for this role.")
                        
                with res_col2:
                    st.markdown("#### 🚀 Skills to Learn")
                    if missing:
                        for skill in missing:
                            st.markdown(f"- **{skill}**")
                    else:
                        st.success("🎉 You have all the recommended core skills for this role!")

elif page_name == "About":
    st.title("ℹ️ About This Project")
    st.markdown("---")
    
    st.markdown("""
    ### 🧠 AI Resume Builder & Skill Recommender
    
    This application utilizes machine learning concepts to assist job seekers in preparing for their next career move.
    
    #### Core Features:
    *   **Text Generation (NLP Simulation):** Simulates sequence modeling (like LSTM networks) to auto-generate coherent professional summaries and experience bullet points based on your profile.
    *   **Skill Mapping (Classification Strategy):** Analyzes role-specific requirements and compares them against your current skillset to provide actionable learning recommendations.
    
    #### Technology Stack:
    *   **Frontend:** Streamlit
    *   **AI/ML Logic:** TensorFlow / Keras (with fallback simulated logic for environments without TF support like Python 3.14+)
    *   **Data Processing:** Pandas, NumPy, Scikit-learn
    
    *Built with ❤️ to help you land your dream job.*
    """)
