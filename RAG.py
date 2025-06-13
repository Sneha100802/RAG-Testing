{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a5b71cc7-0706-4ab3-b40d-55c2344c76a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "import faiss\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "6664cee4-6359-45bd-87d3-cd029ea1e5aa",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-06-12 14:34:03.916 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager\n",
      "2025-06-12 14:34:03.916 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager\n",
      "2025-06-12 14:34:03.916 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-06-12 14:34:04.374 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\jhasn\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-06-12 14:34:04.374 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-06-12 14:34:04.374 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-06-12 14:34:04.400 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-06-12 14:34:04.400 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# --- Load Data ---\n",
    "@st.cache_data\n",
    "def load_data():\n",
    "    df = pd.read_csv( r\"C:\\Users\\jhasn\\OneDrive\\Desktop\\project\\final_startup_data_cleaned.csv\").fillna(\"\")\n",
    "    df['intent_text'] = df['Problem Statement'] + ' ' + df['Advantages'] + ' ' + \\\n",
    "                        df['Target Audience'] + ' ' + df['Tech Stack Required'] + ' ' + \\\n",
    "                        df['Business Model Type'] + ' ' + df['Budget Range']\n",
    "    return df\n",
    "\n",
    "df = load_data()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "deb5964c-95a0-4b1b-9507-53731696cd48",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-06-12 14:34:04.416 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-06-12 14:34:04.416 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-06-12 14:34:04.416 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-06-12 14:34:04.440 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-06-12 14:34:04.441 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# --- TF-IDF + FAISS Setup ---\n",
    "@st.cache_resource\n",
    "def build_index(texts):\n",
    "    vectorizer = TfidfVectorizer()\n",
    "    X = vectorizer.fit_transform(texts)\n",
    "    X_dense = X.toarray().astype(\"float32\")\n",
    "\n",
    "    index = faiss.IndexFlatL2(X_dense.shape[1])\n",
    "    index.add(X_dense)\n",
    "    return vectorizer, index, X_dense\n",
    "\n",
    "vectorizer, index, embeddings = build_index(df['intent_text'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4800e68d-f895-4636-aa87-13f25e4dbfc2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Retrieval Logic ---\n",
    "def get_similar_startups(query, top_k=3):\n",
    "    query_vec = vectorizer.transform([query]).toarray().astype(\"float32\")\n",
    "    _, idx = index.search(query_vec, top_k)\n",
    "    return df.iloc[idx[0]]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "ee2f8049-3b52-4458-a6ea-6e207df344e4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-06-12 14:34:04.452 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-06-12 14:34:04.456 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-06-12 14:34:04.458 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-06-12 14:34:04.458 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-06-12 14:34:04.458 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-06-12 14:34:04.458 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-06-12 14:34:04.458 Session state does not function when running a script without `streamlit run`\n",
      "2025-06-12 14:34:04.458 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-06-12 14:34:04.464 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "# --- UI ---\n",
    "st.title(\"💡 Startup Idea Recommender (Offline, No API)\")\n",
    "query = st.text_input(\"What's your startup idea, goal, or audience?\")\n",
    "\n",
    "if query:\n",
    "    st.write(\"🔍 Finding similar startup ideas...\")\n",
    "    results = get_similar_startups(query)\n",
    "\n",
    "    for _, row in results.iterrows():\n",
    "        st.markdown(f\"### 🚀 {row['Project Name']}\")\n",
    "        st.write(f\"**Problem:** {row['Problem Statement']}\")\n",
    "        st.write(f\"**Tech Stack:** {row['Tech Stack Required']}\")\n",
    "        st.write(f\"**Audience:** {row['Target Audience']}\")\n",
    "        st.write(f\"**Business Model:** {row['Business Model Type']}\")\n",
    "        st.write(f\"**Budget:** {row['Budget Range']}\")\n",
    "        st.markdown(\"---\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
