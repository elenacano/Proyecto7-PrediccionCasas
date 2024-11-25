import streamlit as st
import pandas as pd
import pickle

def load_models():
    with open('modelos/target_encoder.pkl', 'rb') as f:
        target_encoder = pickle.load(f)
    with open('modelos/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('modelos/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return target_encoder, scaler, model

target_encoder, scaler, model = load_models()


# Configuración de la página
st.set_page_config(
    page_title="Predicción de precios de casas",
    page_icon="🏡",
    layout="centered",
)

# Inyectar CSS para cambiar el fondo de la aplicación
st.markdown(
    """
    <style>
    .stApp {
        background-color: #EBDEF0; /* Fondo rosa oscuro */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título y descripción
st.title("Predicción de precios de casas con Machine Learning")
st.write("Usa esta aplicación para saber lo que va a valer tu casa")

# Imagen
st.image(
    "https://fincaraiz.com.co/blog/wp-content/uploads/2022/08/casas-modernas-1-1920x1130.jpg",
    caption="Tu próxima casa",
    use_container_width=True,
)


col1, col2 = st.columns(2)

with col1:
    # Selección del barrio

    municipality = st.selectbox(
        "Municipio",
        [
            'Madrid', 'San Sebastián de los Reyes', 'Villamanrique de Tajo',
            'Rascafría', 'Manzanares el Real', 'Miraflores de la Sierra',
            'Galapagar', 'Arganda', 'San Lorenzo de el Escorial',
            'Aldea del Fresno', 'Aranjuez', 'Villanueva del Pardillo',
            'Las Rozas de Madrid', 'Navalcarnero', 'Alcalá de Henares',
            'El Escorial', 'Leganés', 'Coslada', 'Torrejón de Ardoz',
            'Camarma de Esteruelas', 'Alcorcón', 'Pinto', 'Valdemoro',
            'Collado Villalba', 'Getafe', 'Paracuellos de Jarama', 'El Molar',
            'Parla', 'Tres Cantos', 'Quijorna', 'Valdemorillo', 'Pedrezuela',
            'Daganzo de Arriba', 'Guadarrama', 'Cobeña', 'El Álamo', 'Algete',
            'Rivas-Vaciamadrid', 'Los Santos de la Humosa',
            'San Fernando de Henares', 'Fuenlabrada', 'Mataelpino',
            'Villa del Prado', 'Los Molinos', 'Colmenar Viejo', 'Móstoles',
            'Navalafuente', 'Meco', 'Robledo de Chavela', 'Campo Real',
            'Villaviciosa de Odón', 'Pozuelo de Alarcón', 'Bustarviejo',
            'Collado Mediano', 'Chinchón', 'Colmenarejo', 'Loeches',
            'Sevilla la Nueva', 'Serranillos del Valle', 'Torrelaguna',
            'Villalbilla', 'Alcobendas'
        ]
    )
    st.write(f"Municipio seleccionado: {municipality}")

    property_type = st.selectbox(
        "Tipo de propiedad",
        ["Flat", "Studio", "Duplex", "Other"]
    )
    st.write(f"Tipo de propiedad: {property_type}")

    # Selección del tipo de piso
    floor = st.selectbox(
        "Piso",
        ['3', 'bj', '2', '1', '5', 'en', '4', 'st', '8', '7', '6', '14', 'ss']
    )
    st.write(f"Piso: {floor}")

    # Entrada de la distancia
    distance = st.number_input(
        "Distancia al centro (metros))",
        min_value=0.0,  # Valor mínimo
        step=0.1        # Incremento
    )
    st.write(f"Distancia al centro: {distance} metros")

    exterior = st.selectbox(
        "Exterior",
         ["Sí", "No"]
    )
    st.write(f"Exterior: {exterior}")

    


with col2:
    size = st.number_input(
        "Tamaño (m²)",
        min_value=0.0,  # Valor mínimo
        step=0.1        # Incremento
    )
    st.write(f"Tamaño: {size} m²")

    # Entrada del número de habitaciones
    rooms = st.number_input(
        "Número de habitaciones",
        min_value=1,    # Valor mínimo
        step=1          # Incremento
    )
    st.write(f"Habitaciones: {rooms}")

    # Entrada del número de baños
    bathrooms = st.number_input(
        "Número de baños",
        min_value=1,    # Valor mínimo
        step=1          # Incremento
    )
    st.write(f"Baños: {bathrooms}")

    # Selección del tipo de piso
    has_lift = st.selectbox(
        "¿Tiene ascensor?",
         ["Sí", "No"]
    )
    st.write(f"Ascensor: {has_lift}")


dic_pred = {
    'propertyType':property_type,
    'size': size,
    'exterior': exterior,
    'rooms': rooms,
    'bathrooms' : bathrooms,
    'municipality' : municipality,
    'distance' : distance,
    'floor' : floor,
    'hasLift' : has_lift
}


if st.button("Predecir"):
    df_pred = pd.DataFrame(dic_pred, index=[0])
    st.write(df_pred)

