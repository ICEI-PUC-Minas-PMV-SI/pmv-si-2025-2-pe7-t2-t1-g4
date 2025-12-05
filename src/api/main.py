from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostClassifier
import logging
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # aceita requisição de qualquer front-end
    allow_credentials=True,
    allow_methods=["*"],        # libera GET, POST, PUT, DELETE, etc
    allow_headers=["*"],        # libera headers customizados
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# 1) Carregar o modelo (modelo_final salvo como .cbm)
# -----------------------------
model = CatBoostClassifier()
model.load_model("modelo_catboost_grupo4.cbm")
logger.info("Modelo CatBoost carregado: modelo_catboost_grupo4.cbm")

# -----------------------------
# 2) Schema de entrada (sem product_id — corresponde ao model_final)
# -----------------------------
class PredictRequest(BaseModel):
    ship_mode: str
    customer_id: str
    segment: str
    country: str
    city: str
    state: str
    postal_code: float
    region: str
    category: str
    sub_category: str
    sales: float
    ano: int
    mes: int
    dia_semana: str
    day_of_week: str


@app.get("/")
def root():
    return {"message": "API de previsão operacional (model_final)"}


# -----------------------------
# 3) Ordem das colunas esperada pelo model_final
# -----------------------------
ORDERED_COLS = [
    "ship_mode",
    "customer_id",
    "segment",
    "country",
    "city",
    "state",
    "postal_code",
    "region",
    "category",
    "sub-category",
    "sales",
    "ano",
    "mes",
    "dia_semana",
    "day_of_week",
]

LABEL_MAP = {
    0: "very low",
    1: "low",
    2: "neutral",
    3: "high",
    4: "very high",
}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        # Montar dataframe com nomes de coluna exatamente iguais aos usados no treino
        data = {
            "ship_mode": req.ship_mode,
            "customer_id": req.customer_id,
            "segment": req.segment,
            "country": req.country,
            "city": req.city,
            "state": req.state,
            "postal_code": float(req.postal_code),
            "region": req.region,
            "category": req.category,
            "sub-category": req.sub_category,  # note o hífen: corresponde ao X_train_final
            "sales": float(req.sales),
            "ano": int(req.ano),
            "mes": int(req.mes),
            "dia_semana": req.dia_semana,
            "day_of_week": req.day_of_week,
        }

        df = pd.DataFrame([data])

        # Reproduzir o pré-processamento do treino
        # dia_semana e day_of_week foram convertidos para códigos de categoria no treino
        df["dia_semana"] = df["dia_semana"].astype("category").cat.codes
        df["day_of_week"] = df["day_of_week"].astype("category").cat.codes

        # Forçar a ordem das colunas esperada pelo modelo_final
        df = df[ORDERED_COLS]

        # Previsão
        pred = model.predict(df)
        pred_int = int(pred[0])
        label = LABEL_MAP.get(pred_int, str(pred_int))

        return {"demand_prediction": label}

    except Exception as e:
        logger.exception("Erro durante a predição")
        # Devolver uma mensagem amigável para o Swagger/cliente
        raise HTTPException(status_code=400, detail=f"Erro ao processar a requisição: {e}")


# Exemplo para rodar localmente (comando):
# python -m uvicorn main:app --reload
