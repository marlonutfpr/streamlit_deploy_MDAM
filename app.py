import streamlit as st
import pandas as pd
import joblib

# --- Configuração da Página ---
st.set_page_config(
    page_title="Preditor de Espécie de Íris",
    page_icon="🌸",
    layout="centered"
)

# --- Carregamento do Modelo e Nomes das Classes ---
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Erro: Arquivo do modelo não encontrado em {model_path}")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

@st.cache_data # Usar cache para dados que não mudam (nomes das classes)
def load_class_names(names_path):
    try:
        class_names = joblib.load(names_path)
        return class_names
    except FileNotFoundError:
        st.error(f"Erro: Arquivo de nomes das classes não encontrado em {names_path}")
        return ['Classe 0', 'Classe 1', 'Classe 2'] # Fallback
    except Exception as e:
        st.error(f"Erro ao carregar nomes das classes: {e}")
        return ['Classe 0', 'Classe 1', 'Classe 2'] # Fallback

model = load_model('modelo_iris.joblib')
class_names = load_class_names('nomes_classes_iris.joblib')

# --- Interface do Usuário ---
st.title("🌸 Preditor de Espécie de Íris")
st.markdown("Insira as características da flor para prever a espécie usando um modelo de Regressão Logística.")

st.sidebar.header("Parâmetros de Entrada")

# Coletar inputs do usuário na barra lateral
sepal_length = st.sidebar.slider('Comprimento da Sépala (cm)', min_value=4.0, max_value=8.0, value=5.8, step=0.1)
sepal_width = st.sidebar.slider('Largura da Sépala (cm)', min_value=2.0, max_value=4.5, value=3.0, step=0.1)
petal_length = st.sidebar.slider('Comprimento da Pétala (cm)', min_value=1.0, max_value=7.0, value=4.3, step=0.1)
petal_width = st.sidebar.slider('Largura da Pétala (cm)', min_value=0.1, max_value=2.5, value=1.3, step=0.1)

# Organizar inputs em um DataFrame para o modelo
input_data = pd.DataFrame({
    'sepal length (cm)': [sepal_length],
    'sepal width (cm)': [sepal_width],
    'petal length (cm)': [petal_length],
    'petal width (cm)': [petal_width]
})

st.subheader("Características Inseridas:")
st.write(input_data)

# --- Predição e Exibição ---
if st.button("🌸 Prever Espécie"):
    if model is not None:
        try:
            prediction_idx = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]

            predicted_class_name = class_names[prediction_idx]
            confidence = prediction_proba[prediction_idx] * 100

            st.success(f"**Espécie Prevista:** {predicted_class_name}")
            st.info(f"**Confiança:** {confidence:.2f}%")

            # Exibir probabilidades para todas as classes (opcional)
            st.subheader("Probabilidades por Classe:")
            proba_df = pd.DataFrame({'Classe': class_names, 'Probabilidade': prediction_proba * 100})
            st.bar_chart(proba_df.set_index('Classe'))

        except Exception as e:
            st.error(f"Erro durante a predição: {e}")
    else:
        st.error("Modelo não carregado. Verifique os logs.")

st.sidebar.markdown("---")
st.sidebar.info("Este app usa um modelo treinado no dataset Iris.")
