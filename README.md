# tweet-sentiment-api

API Flask d'analyse de sentiments sur tweets, basee sur deux modeles
de regression logistique scikit-learn, l'un pour la probabilite
positive, l'autre pour la probabilite negative. Le score final,
compris entre -1 et 1, correspond a la difference entre ces deux
probabilites. Le dataset annote est stocke dans une base MySQL.

## Structure

```
tweet-sentiment-api/
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
├── config.py
├── app.py
├── api/          routes.py, errors.py
├── ml/           model.py, training.py, evaluation.py
├── db/           connection.py, queries.py
├── scripts/      setup_database.py, load_sample_data.py, retrain_model.py, setup_cronjob.sh, data/
├── models/       sentiment_model.pkl, model_history/
├── logs/
├── reports/
├── tests/
└── utils/        model_loader.py, paths.py
```

## Prerequis

- Python 3.11+
- MySQL 8
- Docker + Docker Compose si vous voulez lancer MySQL en local sans installation native.

## Installation

```bash
python3 -m venv venv

# macOS / Linux
source venv/bin/activate
# Windows (cmd)
venv\Scripts\activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

pip install -e .
```

## Configuration

```bash
cp .env.example .env      # macOS / Linux
copy .env.example .env    # Windows
```

```env
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=root
MYSQL_DATABASE=sentiment_db
FLASK_DEBUG=True
FLASK_PORT=5050
MODEL_PATH=models/sentiment_model.pkl
```

Faites attention à ce que votre port ne soit pas occupé

## Base de donnees

Avec Docker Compose, lancez MySQL localement avec la meme configuration que l'application:

```bash
docker compose up -d mysql
```

Le service expose MySQL sur `127.0.0.1:3306` et reutilise les variables du fichier `.env`.

Creation de la base et de la table `tweets` :

```bash
python scripts/setup_database.py
```

Sortie attendue :

```
Base 'sentiment_db' prete.
Table 'tweets' prete.
```

Verification de la structure :

```bash
mysql -u root -e "USE sentiment_db; DESCRIBE tweets;"
```

```
+----------+------------+------+-----+---------+----------------+
| Field    | Type       | Null | Key | Default | Extra          |
+----------+------------+------+-----+---------+----------------+
| id       | int        | NO   | PRI | NULL    | auto_increment |
| text     | text       | NO   |     | NULL    |                |
| positive | tinyint(1) | NO   |     | 0       |                |
| negative | tinyint(1) | NO   |     | 0       |                |
+----------+------------+------+-----+---------+----------------+
```

La table est vide a ce stade. Le peuplement se fait via
`scripts/load_sample_data.py`, qui charge un CSV de tweets annotes
verse dans le repo.

## API

### POST /analyze

Accepte un tableau JSON de chaines de caracteres et retourne un objet
associant chaque tweet a son score de sentiment.

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '["Ce produit est incroyable !", "Quelle deception, je ne recommande pas."]'
```

```json
{
  "Ce produit est incroyable !": 0.82,
  "Quelle deception, je ne recommande pas.": -0.65
}
```

### Erreurs

| Cas                              | Code | Message                                                                    |
| -------------------------------- | ---- | -------------------------------------------------------------------------- |
| Liste vide                       | 400  | La liste de tweets ne peut pas etre vide.                                  |
| Corps non-liste                  | 400  | Le corps de la requete doit etre un tableau JSON de chaines de caracteres. |
| Element non-string dans la liste | 400  | Chaque element du tableau doit etre une chaine de caracteres.              |
| Modele absent                    | 503  | Le modele de sentiment n'est pas disponible.                               |

```bash
curl -X POST http://localhost:5000/analyze -H "Content-Type: application/json" -d '[]'
# {"error": "La liste de tweets ne peut pas etre vide."}
```

```bash
curl -X POST http://localhost:5000/analyze -H "Content-Type: application/json" -d '{"tweet": "test"}'
# {"error": "Le corps de la requete doit etre un tableau JSON de chaines de caracteres."}
```

## Reentrainement du modele

_A completer par Personne 3 : configuration du cronjob macOS et de
l'equivalent Task Scheduler sous Windows pour `scripts/retrain_model.py`._
