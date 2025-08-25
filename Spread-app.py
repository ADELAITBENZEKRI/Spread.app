import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.interpolate import interp1d
from datetime import datetime
import plotly.graph_objects as go
###
# Ajoutez cette section après vos imports existants
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')
import time
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from plotly.subplots import make_subplots
###

#######
# Configuration de la page
st.set_page_config(page_title="Analyse des Spreads Obligataires Maroc", layout="wide")

# Titre de l'application
st.title(" Analyse Interactive des Spreads Obligataires Marocains")

# Définition des constantes
MATURITY_LABELS = ['<1 an', '1-2 ans', '2-3 ans', '3-5 ans', '5-7 ans', '7-10 ans', '10-15 ans', '15-20 ans', '20-30 ans', '30+ ans']

# Initialisation des variables de session
if 'bonds_data' not in st.session_state:
    st.session_state.bonds_data = None
if 'rates_data' not in st.session_state:
    st.session_state.rates_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'spreads_calculated' not in st.session_state:
    st.session_state.spreads_calculated = False
if 'sector_added' not in st.session_state:
    st.session_state.sector_added = False
if 'show_abbreviations' not in st.session_state:
    st.session_state.show_abbreviations = False

# Barre d'upload de données
st.header("1. Chargement des données")
uploaded_file = st.file_uploader("Chargez le fichier Excel (contenant les feuilles 'BONDS' et 'HISTOR_TAUX')", type=["xlsx"])

# Fonctions de traitement des données
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            bonds_data = pd.read_excel(uploaded_file, sheet_name="BONDS")
            rates_data = pd.read_excel(uploaded_file, sheet_name="HISTOR_TAUX")
            st.session_state.bonds_data = bonds_data
            st.session_state.rates_data = rates_data
            st.session_state.processed_data = None
            st.session_state.spreads_calculated = False
            st.session_state.sector_added = False
            st.success("Données chargées avec succès!")
            return True
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {str(e)}")
            return False
    else:
        return False

def preprocess_data():
    try:
        bonds_data = st.session_state.bonds_data.copy()
        rates_data = st.session_state.rates_data.copy()
        
        # Nettoyage des données obligataires
        bonds_data['ISSUEDT'] = pd.to_datetime(bonds_data['ISSUEDT'], errors='coerce')
        bonds_data['MATURITYDT_L'] = pd.to_datetime(bonds_data['MATURITYDT_L'], errors='coerce')
        bonds_data['DAYS_TO_MATURITY'] = (bonds_data['MATURITYDT_L'] - bonds_data['ISSUEDT']).dt.days
        bonds_data['INTERESTRATE'] = pd.to_numeric(bonds_data['INTERESTRATE'], errors='coerce') / 100
        
        # Nettoyage des taux souverains
        rates_data['DATE'] = pd.to_datetime(rates_data['DATE'], errors='coerce')
        rates_data = rates_data.sort_values('DATE').dropna(subset=['DATE'])
        
        st.session_state.bonds_data = bonds_data
        st.session_state.rates_data = rates_data
        st.session_state.processed_data = bonds_data.copy()
        st.success("Prétraitement des données terminé avec succès!")
        return True
    except Exception as e:
        st.error(f"Erreur lors du prétraitement des données : {str(e)}")
        return False


##########calcul des spreads
def calculate_spreads():
    try:
        bonds_data = st.session_state.bonds_data.copy()
        rates_data = st.session_state.rates_data.copy()
        
        # Configuration des maturités
        maturity_days = {
            '13 Semaines': 91, '26 Semaines': 182, '52 Semaines': 364,
            '2 Ans': 730, '5 Ans': 1825, '10 ans': 3650,
            '15 ans': 5475, '20 ans': 7300, '30 ans': 10950
        }
        
        # Interpolation des taux souverains
        days_range = np.arange(91, 10951)
        valid_dates = rates_data['DATE'].dropna().unique()
        interpolated_rates = pd.DataFrame(index=valid_dates, columns=[f'J{day}' for day in days_range])
        
        for date in valid_dates:
            date_rates = rates_data[rates_data['DATE'] == date].iloc[0]
            known_days = []
            known_rates = []
            
            for col, days in maturity_days.items():
                rate = date_rates.get(col, np.nan)
                if not pd.isna(rate):
                    known_days.append(days)
                    known_rates.append(rate)
            
            if len(known_days) >= 2:
                interpolated_rates.loc[date] = np.interp(days_range, known_days, known_rates)
            else:
                st.warning(f"Données insuffisantes pour interpolation à la date {date}")
        
        # Filtrer les lignes avec des dates ou maturités invalides
        bonds_data = bonds_data.dropna(subset=['ISSUEDT', 'DAYS_TO_MATURITY'])
        bonds_data = bonds_data[bonds_data['ISSUEDT'].notna()]
        
        # Calcul des spreads
        def calculate_spread(row):
            try:
                days = int(row['DAYS_TO_MATURITY'])
                if days not in days_range:
                    return np.nan
                asof_date = interpolated_rates.index.asof(row['ISSUEDT'])
                if pd.isna(asof_date):
                    return np.nan
                if row['ISSUEDT'] > interpolated_rates.index.max():
                    return np.nan
                return row['INTERESTRATE'] - interpolated_rates.loc[asof_date, f'J{days}']
            except Exception as e:
                st.warning(f"Erreur pour l'émetteur {row.get('PREFERREDNAMEISSUER', 'Inconnu')} : {str(e)}")
                return np.nan
        
        bonds_data['SPREAD'] = bonds_data.apply(calculate_spread, axis=1)
        bonds_data['SPREAD_POINTS'] = bonds_data['SPREAD'] * 10000
        
        st.session_state.processed_data = bonds_data.dropna(subset=['SPREAD'])
        st.session_state.spreads_calculated = True
        st.success("Calcul des spreads terminé avec succès!")
        return True
    except Exception as e:
        st.error(f"Erreur lors du calcul des spreads : {str(e)}")
        return False

def add_sector_classification():
    try:
        bonds_data = st.session_state.processed_data.copy()
        
        # Classification sectorielle
        public = ['FEC', 'CDG K E']
        banques = ['BMCI', 'BOA', 'SGMB', 'CDM', 'ATW E', 'CFG BANK', 'CIH E', 
                  'AL BARID BANK E', 'BCP E', 'CAM E']
        societes_financement = ['SALAFIN', 'TASLIF', 'MAGHREBAIL', 'WAFASALAF', 
                               'WAFABAIL', 'SOGELEASE', 'BMCI LEASI', 'AXA CREDIT', 
                               'SOFAC CREDIT', 'RCI', 'JAIDA', 'MA LEASING', 
                               'VIVALIS SALAF', 'FINAN HATT', 'MGT III', 'MGT TITRIS']
        assurance = ['SAHAM']
        societes_investissement = ['GROUP INVEST SA', 'SAHAM FINANCES', 'HOLMARCOM',
                                  'FINANCECOM', 'AL MADA', 'O CAPITAL GROUP', 
                                  'FINANCIER SEC']
        immobilier = ['DOUJA PROM ADD', 'IMMOLOG', 'CGI', 'CAPEP', 'BEST REAL ESTAT',
                     'DYAR AL MANSOUR', 'AL OMRANE', 'ADI', 'ALLIANCES DARNA'] + \
                    [f'FT DOMUS {i}' for i in range(1, 17)] + \
                    ['MGT IX CONS', 'MGT IV', 'MGT VI IMMO', 'ARADEI CAPITAL',
                     'MGT SAKANE', 'MGT VIII FT', 'GARAN']
        tourisme = ['RISMA SA']
        distribution_commerce = ['LABEL VIE', 'UNIVERS MOTORS', 'DISTRA-S.A', 
                               'DISWAY', 'MARJANE HOLDING', 'SEDM']
        industrie = ['HOLCIM MAROC', 'MAGHREB STEEL', 'NEXANS MAROC', 'CIMAT', 
                    'SETTAVEX', 'MAGHREB OXYGENE', 'MANAGEM', 'JET CONTRACTORS', 
                     'UNIMER', 'TGCC']
        agroalimentaire = ['MUTANDIS SCA', 'ZALAGH HOLDING', 
                           'OULMES']
        energie = ['SAMIR', 'MGT IV ENER I', 'MGT II ENER II' ,'AFRIQUIA GAZ']
        infrastructures = ['ADM', 'ANP', 'TANGER MED SA', 'ONCF', 
                         'TMPA', 'COMANAV', 'NADOR WEST MED']
        services_public = ['LYDEC', 'MASEN']
        telecoms = ['MEDI TELCOM SA', 'MEDIACO MAROC']
        services = ['VALYANS', 'RDS', 'PALME DEV', 'FT SYNTHESIUM', 
                  'FT RELEVIUM I', 'FINANCIER SEC']

        sector_mapping = {}
        for secteur, emetteurs in [
            ('Publique', public),
            ('Banques', banques),
            ('Sociétés de financement/Leasing/Crédit', societes_financement),
            ('Assurance', assurance),
            ('Sociétés d\'investissement/Holding', societes_investissement),
            ('Immobilier & Aménagement', immobilier),
            ('Tourisme', tourisme),
            ('Distribution & Commerce', distribution_commerce),
            ('Industrie & Matériaux', industrie),
            ('Agroalimentaire', agroalimentaire),
            ('Énergie & Environnement', energie),
            ('Infrastructures & Transport', infrastructures),
            ('Services Public', services_public),
            ('Télécoms', telecoms),
            ('Services/Informatique/Conseil', services)
        ]:
            for emetteur in emetteurs:
                sector_mapping[emetteur] = secteur
        
        bonds_data['SECTEUR'] = bonds_data['PREFERREDNAMEISSUER'].map(sector_mapping).fillna('Divers')
        
        # Ajout des buckets de maturité
        bonds_data['MATURITY_YEARS'] = bonds_data['DAYS_TO_MATURITY'] / 365
        maturity_bins = [0] + list(range(1, 31)) + [float('inf')]
        maturity_labels = [f"{i}-{i+1}" for i in range(30)] + ['30+']

        bonds_data['MATURITY_BUCKET'] = pd.cut(
            bonds_data['MATURITY_YEARS'], 
            bins=maturity_bins,
            labels=maturity_labels,
            right=False,
            include_lowest=True
        )
        
        st.session_state.processed_data = bonds_data
        st.session_state.sector_added = True
        st.success("Classification sectorielle ajoutée avec succès!")
        return True
    except Exception as e:
        st.error(f"Erreur lors de la classification sectorielle : {str(e)}")
        return False
#######image###########        
st.sidebar.image("logo ALBARID.png", width=300)
# Bouton pour ouvrir le panneau des abréviations
if st.sidebar.button("📚 Afficher les abréviations"):
    st.session_state.show_abbreviations = not st.session_state.show_abbreviations

# Panneau des abréviations
if st.session_state.show_abbreviations:
    with st.sidebar.expander("📖 Abréviations BONDS", expanded=True):
        st.markdown("""
        # Abréviations dans la feuille BONDS

        Liste des abréviations trouvées dans la feuille *BONDS* du fichier Excel, avec leurs significations probables.

        ---

        ## 📌 Colonnes principales

        | Abréviation           | Signification probable                          |
        |-----------------------|-------------------------------------------------|
        | INSTRID             | Identifiant de l'instrument                     |
        | INSTRTYPE           | Type d'instrument (ex: FRBD, FLRT, AMBD, etc.)  |
        | INSTRCTGRY          | Catégorie d'instrument (ex: OBL_ORDN, OBL_SUBD) |
        | ENGPREFERREDNAME    | Nom préféré en anglais                          |
        | ENGLONGNAME         | Nom long en anglais                             |
        | ISSUERCD            | Code de l'émetteur                              |
        | ISSUECAPITAL        | Capital émis                                    |
        | ISSUESIZE           | Taille de l'émission                            |
        | ISSUEDT             | Date d'émission                                 |
        | MATURITYDT_L        | Date d'échéance                                 |
        | DAYS_TO_MATURITY    | Jours jusqu'à l'échéance                        |
        | PARVALUE            | Valeur nominale                                 |
        | INTERESTTYPE        | Type d'intérêt (FIXD, FLOT, etc.)               |
        | FORM                | Forme (BR, RGD, etc.)                           |
        | GUARANTEE           | Garantie                                        |
        | NEWPARVALUE         | Nouvelle valeur nominale                        |
        | INTERESTPERIODCTY   | Périodicité des intérêts (ANLY, HFLY, QTLY)     |
        | INTERESTRATE        | Taux d'intérêt                                  |
        | REDEMPTIONTYPE      | Type de remboursement (INST, BLET, AMOR)        |
        | PREFERREDNAMEREGISTRAR | Nom du registraire préféré                  |
        | PREFERREDNAMEISSUER | Nom de l'émetteur préféré                       |
        | INSTRSTATUS         | Statut de l'instrument (ACTI, DLET)             |

        ---

        ## 🔤 Types d'instruments (INSTRTYPE)

        | Abréviation | Signification complète                         |
        |------------|-----------------------------------------------|
        | FRBD       | Fixed Rate Bond Debt (Obligation à taux fixe) |
        | FLRT       | Floating Rate (Taux variable)                 |
        | AMBD       | Amortizing Bond Debt (Obligation amortissable)|
        | ZCBD       | Zero Coupon Bond Debt (Coupon zéro)           |
        | TCN        | Titre de Créance Négociable                   |

        ---

        ## 🧩 Catégories d'instruments (INSTRCTGRY)

        | Abréviation | Signification                     |
        |------------|-----------------------------------|
        | OBL_ORDN   | Obligation ordinaire              |
        | OBL_SUBD   | Obligation subordonnée            |
        | OBL_CONV   | Obligation convertible            |
        | FPCT       | Fonds de Placement en Créances Titrisées |
        | TCN        | Titre de Créance Négociable       |

        ---

        ## 💹 Types d'intérêt (INTERESTTYPE)

        | Abréviation | Signification           |
        |------------|-------------------------|
        | FIXD       | Fixed (Taux fixe)       |
        | FLOT       | Floating (Taux variable)|
        | FLTG       | Floating (variante)     |
        | DISC       | Discount (Escompte)     |

        ---

        ## 📄 Formes (FORM)

        | Abréviation | Signification           |
        |------------|-------------------------|
        | BR         | Bearer (Au porteur)     |
        | RGD        | Registered (Nominatif)  |

        ---

        ## 🛡 Garanties (GUARANTEE)

        | Abréviation | Signification           |
        |------------|-------------------------|
        | GTG        | Garantie                |
        | USUG       | Sans garantie (probable)|

        ---

        ## 🔄 Périodicité des intérêts (INTERESTPERIODCTY)

        | Abréviation | Signification     |
        |------------|-------------------|
        | ANLY       | Annuel            |
        | HFLY       | Semestriel        |
        | QTLY       | Trimestriel       |

        ---

        ## 💳 Types de remboursement (REDEMPTIONTYPE)

        | Abréviation | Signification                              |
        |------------|-------------------------------------------|
        | INST       | In fine (remboursement en une fois)       |
        | BLET       | Bullet (équivalent à in fine)             |
        | AMOR       | Amortissement (remboursement progressif)  |

        ---

        ## 🏁 Statuts (INSTRSTATUS)

        | Abréviation | Signification             |
        |------------|---------------------------|
        | ACTI       | Actif                     |
        | DLET       | Délété (supprimé / échu)  |

        ---

        ## 🏢 Autres abréviations fréquentes dans les noms

        | Abréviation | Signification complète                          |
        |------------|--------------------------------------------------|
        | Ob         | Obligation                                       |
        | ONCF       | Office National des Chemins de Fer               |
        | ATW        | Attijariwafa Bank                                |
        | BMCE       | Banque Marocaine du Commerce Extérieur           |
        | CDM        | Crédit du Maroc                                  |
        | LYDEC      | Lyonnaise des Eaux de Casablanca                 |
        | BOA        | Bank of Africa                                   |
        | OCP        | Office Chérifien des Phosphates                  |
        | ADM        | Autoroutes du Maroc                              |
        | SGMB       | Société Générale Marocaine de Banques            |
        | CDG        | Caisse de Dépôt et de Gestion                    |
        """)

# Boutons pour chaque étape du traitement
if uploaded_file is not None:
    if st.button("1. Charger les données"):
        if load_data(uploaded_file):
            st.subheader("Aperçu des données BONDS")
            st.dataframe(st.session_state.bonds_data.head())
            st.subheader("Résumé des données BONDS")
            st.write(st.session_state.bonds_data.describe(include='all'))
            
            st.subheader("Aperçu des données HISTOR_TAUX")
            st.dataframe(st.session_state.rates_data.head())
            st.subheader("Résumé des données HISTOR_TAUX")
            st.write(st.session_state.rates_data.describe(include='all'))

if st.session_state.bonds_data is not None and st.button("2. Prétraiter les données"):
    if preprocess_data():
        st.subheader("Données BONDS après prétraitement")
        st.dataframe(st.session_state.bonds_data.head())
        st.subheader("Résumé après prétraitement")
        st.write(st.session_state.bonds_data.describe(include='all'))
        
        st.subheader("Données HISTOR_TAUX après prétraitement")
        st.dataframe(st.session_state.rates_data.head())
        st.subheader("Résumé après prétraitement")
        st.write(st.session_state.rates_data.describe(include='all'))

if st.session_state.processed_data is not None and st.button("3. Calculer les spreads"):
    if calculate_spreads():
        st.subheader("Données avec spreads calculés")
        st.dataframe(st.session_state.processed_data.head())
        st.subheader("Résumé des spreads")
        st.write(st.session_state.processed_data[['SPREAD', 'SPREAD_POINTS']].describe())

if st.session_state.spreads_calculated and st.button("4. Ajouter la classification sectorielle"):
    if add_sector_classification():
        st.subheader("Données finales avec classification sectorielle")
        st.dataframe(st.session_state.processed_data.head())
        st.subheader("Répartition par secteur")
        st.write(st.session_state.processed_data['SECTEUR'].value_counts())
        st.subheader("Répartition par maturité")
        st.write(st.session_state.processed_data['MATURITY_BUCKET'].value_counts())

# Section d'analyse (uniquement si toutes les étapes sont complétées)
if st.session_state.sector_added:
    st.header("Analyse des données")
    
    # Sidebar avec les filtres
    st.sidebar.header("Filtres d'analyse")
    filtered_data = st.session_state.processed_data.copy()

    # Filtres de base
    selected_sectors = st.sidebar.multiselect(
        "Secteurs",
        options=sorted(filtered_data['SECTEUR'].unique()),
        default=['Banques']
    )

    selected_instruments = st.sidebar.multiselect(
        "Types d'instruments",
        options=sorted(filtered_data['INSTRCTGRY'].unique()),
        default=['TCN']
    )

    # Application des premiers filtres (secteur et type d'instrument)
    filter_conditions = [
        filtered_data['SECTEUR'].isin(selected_sectors),
        filtered_data['INSTRCTGRY'].isin(selected_instruments)
    ]
    filtered_data = filtered_data[np.logical_and.reduce(filter_conditions)]

    # Filtre conditionnel pour le type de TCN
    if 'TCN' in selected_instruments:
        tcn_data = filtered_data[filtered_data['INSTRCTGRY'] == 'TCN']
        if 'TYPETCN' in tcn_data.columns and not tcn_data['TYPETCN'].isnull().all():
            tcn_types = sorted(tcn_data['TYPETCN'].dropna().unique())
            selected_tcn_types = st.sidebar.multiselect(
                "Type de TCN",
                options=tcn_types,
                default=['CD '] if 'CD ' in tcn_types else tcn_types[:1]
            )
            # Appliquer le filtre TCN
            filtered_data = filtered_data[
                (filtered_data['INSTRCTGRY'] != 'TCN') | 
                (filtered_data['TYPETCN'].isin(selected_tcn_types))
            ]

    # Déterminer les maturités disponibles après filtrage
    available_maturities = sorted(
        filtered_data['MATURITY_BUCKET'].unique(),
        key=lambda x: MATURITY_LABELS.index(x) if x in MATURITY_LABELS else len(MATURITY_LABELS)
    )

    # Définir les maturités par défaut en fonction des instruments
    default_maturities = []
    if 'TCN' in selected_instruments and '0-1' in available_maturities:
        default_maturities.append('0-1')
    if any(inst in selected_instruments for inst in ['OBLIG-SUBD', 'OBLIG-ORDN']) and '10-11' in available_maturities:
        default_maturities.append('10-11')

    # Si pas de maturité par défaut trouvée, prendre les premières disponibles
    if not default_maturities and available_maturities:
        default_maturities = available_maturities[:1]

    selected_maturities = st.sidebar.multiselect(
        "Maturités",
        options=available_maturities,
        default=default_maturities if default_maturities else None
    )

    # Appliquer le filtre des maturités
    if selected_maturities:
        filtered_data = filtered_data[filtered_data['MATURITY_BUCKET'].isin(selected_maturities)]

    # Filtre de périodicité
    available_periodicities = sorted(filtered_data['INTERESTPERIODCTY'].dropna().unique())
    selected_periodicity = st.sidebar.multiselect(
        "Périodicité des intérêts",
        options=available_periodicities,
        default=['ANLY'] if 'ANLY' in available_periodicities else available_periodicities[:1]
    )

    # Appliquer le filtre de périodicité
    if selected_periodicity:
        filtered_data = filtered_data[filtered_data['INTERESTPERIODCTY'].isin(selected_periodicity)]

    selected_instrtype = st.sidebar.multiselect(
    "Type d'instrument",
    options=sorted(filtered_data['INSTRTYPE'].dropna().unique()),
    default=['FRBD']
) if selected_instruments else None

    selected_interest_type = st.sidebar.multiselect(
        "Type d'intérêt",
        options=sorted(filtered_data['INTERESTTYPE'].dropna().unique()),
        default=['FIXD']
    )

    # Filtrage des données avec les nouveaux filtres
    filtered_data = filtered_data[
        (filtered_data['SECTEUR'].isin(selected_sectors)) &
        (filtered_data['INSTRCTGRY'].isin(selected_instruments)) &
        (filtered_data['MATURITY_BUCKET'].isin(selected_maturities)) &
        (filtered_data['INTERESTPERIODCTY'].isin(selected_periodicity)) &
        (filtered_data['INTERESTTYPE'].isin(selected_interest_type))&
        (filtered_data['INSTRTYPE'].isin(selected_instrtype))
    ]

    # Onglets pour différentes visualisations
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📉 Visualisation des Spreads par Maturité","📈 Évolution des spreads par émetteur", "📊 Cartographie Stratégique des Secteurs", "🧩 Exploration Avancée des Instruments", "🔍 Détails", "🔮 Prédiction", "🎯 Modélisation du Spread - Comparaison de Modèles ML", "🏆 Scoring Sectoriel des Émetteurs"])
 #####################333  

    with tab1:
        st.header("📉 Visualisation des Taux et Spreads")
        
        # Vérification des données
        if not st.session_state.spreads_calculated:
            st.error("Veuillez d'abord calculer les spreads dans l'étape 3.")
            st.stop()
        
        # Données filtrées
        vis_data = filtered_data.copy()
        
        if vis_data.empty:
            st.warning("Aucune donnée disponible avec les filtres actuels.")
            st.stop()
        
        # Conversion des taux en pourcentage pour l'affichage
        vis_data['FACIAL_RATE'] = vis_data['INTERESTRATE'] * 100
        vis_data['SOVEREIGN_RATE'] = (vis_data['INTERESTRATE'] - vis_data['SPREAD']) * 100
        vis_data['SPREAD_POINTS'] = vis_data['SPREAD_POINTS'].fillna(0)
        
        # Agrégation des données par date d'émission (on ne groupe plus par maturité)
        agg_data = vis_data.groupby('ISSUEDT').agg({
            'FACIAL_RATE': 'mean',
            'SOVEREIGN_RATE': 'mean',
            'MATURITY_BUCKET': 'first'  # On garde la maturité pour l'affichage
        }).reset_index()
        
        if not agg_data.empty:
            # Récupération de la maturité sélectionnée (première valeur trouvée)
            maturity = agg_data['MATURITY_BUCKET'].iloc[0] if 'MATURITY_BUCKET' in agg_data.columns else "Maturité sélectionnée"
            
            st.subheader(f"Maturité : {maturity}")
            
            # Création du graphique simple sans axe secondaire
            fig = go.Figure()
            
            # Courbe des taux faciaux moyens
            fig.add_trace(go.Scatter(
                x=agg_data['ISSUEDT'],
                y=agg_data['FACIAL_RATE'],
                name='Taux Facial (Privé)',
                line=dict(color='blue', width=2),
                mode='lines+markers'
            ))
            
            # Courbe des taux souverains moyens
            fig.add_trace(go.Scatter(
                x=agg_data['ISSUEDT'],
                y=agg_data['SOVEREIGN_RATE'],
                name='Taux Souverain',
                line=dict(color='green', width=2),
                mode='lines+markers'
            ))
            
            # Zone ombrée pour le spread
            fig.add_trace(go.Scatter(
                x=agg_data['ISSUEDT'].tolist() + agg_data['ISSUEDT'].tolist()[::-1],
                y=agg_data['FACIAL_RATE'].tolist() + agg_data['SOVEREIGN_RATE'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 165, 0, 0.2)',
                line=dict(width=0),
                name='Spread',
                hoverinfo='skip'
            ))
            
            # Mise en forme du graphique
            fig.update_layout(
                title=f"Évolution des Taux - {maturity}",
                xaxis_title="Date d'émission",
                yaxis_title="Taux (%)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode="x unified",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques descriptives
            with st.expander(f"Statistiques pour {maturity}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Taux Faciaux**")
                    st.write(agg_data['FACIAL_RATE'].describe().to_frame().T.style.format("{:.2f}"))
                
                with col2:
                    st.markdown("**Taux Souverains**")
                    st.write(agg_data['SOVEREIGN_RATE'].describe().to_frame().T.style.format("{:.2f}"))
                
                # Calcul des spreads moyens pour les stats
                agg_data['SPREAD_POINTS'] = (agg_data['FACIAL_RATE'] - agg_data['SOVEREIGN_RATE']) * 100
                st.markdown("**Spreads (points)**")
                st.write(agg_data['SPREAD_POINTS'].describe().to_frame().T.style.format("{:.2f}"))
                
                # Téléchargement des données
                csv = agg_data[['ISSUEDT', 'FACIAL_RATE', 'SOVEREIGN_RATE', 'SPREAD_POINTS']].to_csv(index=False)
                st.download_button(
                    label=f"Télécharger les données ({maturity})",
                    data=csv.encode('utf-8'),
                    file_name=f"spreads_{maturity}.csv",
                    mime='text/csv'
                )
        else:
            st.info("Aucune donnée disponible pour les filtres sélectionnés.")





    with tab2:
        st.header("Évolution des spreads par émetteur")
        
        # Nouvelle section : Recherche par INSTRID
        with st.expander("🔍 Recherche par identifiant d'instrument", expanded=True):
            col_search1, col_search2 = st.columns(2)
            
            with col_search1:
                instr_id = st.text_input(
                    "Entrez l'INSTRID (ex: MA1234567890)",
                    placeholder="Identifiant complet ou partiel",
                    help="Recherche par identifiant d'instrument"
                )
            
            with col_search2:
                if st.button("Rechercher", key="search_instr"):
                    if instr_id:
                        search_results = filtered_data[
                            filtered_data['INSTRID'].str.contains(instr_id, case=False, na=False)
                        ]
                        if not search_results.empty:
                            st.session_state['search_results'] = search_results
                        else:
                            st.warning("Aucun instrument trouvé avec cet identifiant")
                            st.session_state['search_results'] = None
                    else:
                        st.warning("Veuillez entrer un identifiant")
        
        # Affichage des résultats de recherche
        if 'search_results' in st.session_state and st.session_state['search_results'] is not None:
            with st.container(border=True):
                st.subheader("Résultats de la recherche")
                res = st.session_state['search_results'][[
                    'INSTRID', 'PREFERREDNAMEISSUER', 'ISSUEDT', 
                    'SPREAD_POINTS', 'DAYS_TO_MATURITY'
                ]].sort_values('ISSUEDT', ascending=False)
                
                # Premier cadre horizontal - Résumé
                with st.container(border=True):
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Spread", f"{res.iloc[0]['SPREAD_POINTS']:,.0f} pb")
                    with cols[1]:
                        st.metric("Émetteur", res.iloc[0]['PREFERREDNAMEISSUER'])
                    with cols[2]:
                        st.metric("Date émission", res.iloc[0]['ISSUEDT'].strftime('%d/%m/%Y'))
                    with cols[3]:
                        st.metric("Maturité", f"{res.iloc[0]['DAYS_TO_MATURITY']} jours")
                
                # Deuxième cadre horizontal - Détails complets
                with st.container(border=True):
                    st.dataframe(
                        res,
                        column_config={
                            "SPREAD_POINTS": st.column_config.NumberColumn(format="%.0f pb"),
                            "DAYS_TO_MATURITY": st.column_config.NumberColumn(format="%.0f jours")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
        # Création de deux colonnes pour une meilleure organisation
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Sélection de plusieurs émetteurs avec option "Tous"
            all_issuers = sorted(filtered_data['PREFERREDNAMEISSUER'].unique())
            selected_issuers = st.multiselect(
                "Sélectionnez un ou plusieurs émetteurs",
                options=all_issuers,
                default=all_issuers[0] if len(all_issuers) > 0 else None
            )
        
        with col2:
            # Sélection de la période d'analyse
            period = st.selectbox(
                "Période d'analyse",
                options=["3 ans", "5 ans", "10 ans", "Toute la période"],
                index=2
            )
        
        # Filtrage des données selon la sélection
        if len(selected_issuers) > 0:
            issuer_data = filtered_data[filtered_data['PREFERREDNAMEISSUER'].isin(selected_issuers)]
        else:
            issuer_data = filtered_data.copy()
            selected_issuers = ["Tous les émetteurs"]
        
        # Application du filtre temporel
        if period != "Toute la période":
            years = int(period.split()[0])
            cutoff_date = pd.to_datetime('today') - pd.DateOffset(years=years)
            issuer_data = issuer_data[issuer_data['ISSUEDT'] >= cutoff_date]
        
        if not issuer_data.empty:
            # Section Principale - Graphique
            fig = px.scatter(
                issuer_data,
                x='ISSUEDT',
                y='SPREAD_POINTS',
                color='PREFERREDNAMEISSUER',  # Couleur par émetteur
                symbol='INSTRCTGRY',  # Symbole par catégorie d'instrument
                facet_col='MATURITY_BUCKET' if len(selected_issuers) > 1 else None,
                title=f"Évolution des spreads pour {', '.join(selected_issuers) if len(selected_issuers) <= 3 else f'{len(selected_issuers)} émetteurs'}",
                labels={'SPREAD_POINTS': 'Spread (points)', 'ISSUEDT': 'Date d\'émission'},
                hover_data=['DAYS_TO_MATURITY', 'INSTRID', 'INTERESTPERIODCTY', 'INTERESTTYPE'],
                trendline="lowess" if len(issuer_data) > 10 else None
            )
            
            # Améliorations du graphique
            fig.update_layout(
                hovermode='closest',
                xaxis_title='Date d\'émission',
                yaxis_title='Spread (points de base)',
                legend_title='Émetteur',
                height=600 if len(selected_issuers) > 1 else 400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Section Secondaire - Métriques et Analyses
            st.subheader("Analyse des spreads")
            
            # Création de 3 colonnes pour les KPI
            kpi1, kpi2, kpi3 = st.columns(3)
            
            with kpi1:
                avg_spread = issuer_data['SPREAD_POINTS'].mean()
                st.metric("Spread moyen", f"{avg_spread:,.0f} pb", 
                        delta=f"{(avg_spread - issuer_data['SPREAD_POINTS'].quantile(0.25)):,.0f} pb vs Q1")
            
            with kpi2:
                volatility = issuer_data['SPREAD_POINTS'].std()
                st.metric("Volatilité des spreads", f"{volatility:,.0f} pb",
                        delta_color="inverse")
            
            with kpi3:
                latest_issue = issuer_data.sort_values('ISSUEDT', ascending=False).iloc[0]
                st.metric("Dernière émission", 
                        f"{latest_issue['SPREAD_POINTS']:,.0f} pb",
                        f"{latest_issue['ISSUEDT'].strftime('%d/%m/%Y')}")
        
        else:
            st.warning("Aucune donnée disponible pour cette sélection.")
            st.info("Essayez d'élargir vos critères de sélection ou de choisir une autre période.")
    
    with tab3:
        st.header("🌐 Cartographie Stratégique des Secteurs")
        
        # =================================================================
        # STYLE ET SCRIPT POUR LES POPUPS
        # =================================================================
        st.markdown("""
        <style>
        /* Bouton d'aide */
        .help-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #f0f2f6;
            color: #1e3c72;
            border: 1px solid #1e3c72;
            margin-left: 8px;
            cursor: pointer;
            font-weight: bold;
            vertical-align: middle;
        }
        
        /* Conteneur popup */
        .popup-container {
            position: relative;
            display: inline-block;
        }
        
        /* Contenu popup */
        .popup-content {
            visibility: hidden;
            position: absolute;
            z-index: 1000;
            right: 0;
            width: 280px;
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        /* Affichage du popup */
        .show {
            visibility: visible;
            opacity: 1;
        }
        
        /* Style du contenu */
        .popup-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 8px;
            color: #1e3c72;
        }
        
        .popup-list {
            padding-left: 20px;
            margin: 0;
            font-size: 14px;
        }
        
        .popup-list li {
            margin-bottom: 6px;
        }
        </style>
        
        <script>
        function togglePopup(elementId) {
            var popup = document.getElementById(elementId);
            popup.classList.toggle("show");
            
            // Fermer les autres popups
            document.querySelectorAll('.popup-content').forEach(function(item) {
                if (item.id !== elementId) {
                    item.classList.remove("show");
                }
            });
        }
        
        // Fermer en cliquant à l'extérieur
        document.addEventListener('click', function(event) {
            if (!event.target.matches('.help-btn') && !event.target.closest('.popup-content')) {
                document.querySelectorAll('.popup-content').forEach(function(item) {
                    item.classList.remove("show");
                });
            }
        });
        </script>
        """, unsafe_allow_html=True)

        # =================================================================
        # SECTION 1 : ANALYSE THERMIQUE
        # =================================================================
        col_title, col_help = st.columns([0.9, 0.1])
        with col_title:
            st.subheader("-> Rendements Sectoriels")
        with col_help:
            st.markdown("""
            <div class="popup-container">
                <div class="help-btn" onclick="togglePopup('heatmap-help')">?</div>
                <div id="heatmap-help" class="popup-content">
                    <div class="popup-title">Guide d'interprétation</div>
                    <ul class="popup-list">
                        <li><b>Couleur</b> : Du rouge (risque élevé) au vert (risque faible)</li>
                        <li><b>Lignes</b> : Périodes chronologiques</li>
                        <li><b>Colonnes</b> : Secteurs d'activité</li>
                        <li><b>Insight</b> : Les zones rouges persistantes indiquent des stress sectoriels</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualisation thermique
        heat_data = filtered_data.pivot_table(
            index=filtered_data['ISSUEDT'].dt.strftime('%Y-%m'),
            columns='SECTEUR',
            values='SPREAD_POINTS',
            aggfunc='mean'
        )
        fig = px.imshow(
            heat_data,
            color_continuous_scale='RdYlGn_r',
            labels=dict(x="Secteur", y="Période", color="Spread"),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        # =================================================================
        # SECTION 2 : COMPARAISON SECTORIELLE
        # =================================================================
        col_title, col_help = st.columns([0.9, 0.1])
        with col_title:
            st.subheader("📊 Benchmark Sectoriel")
        with col_help:
            st.markdown("""
            <div class="popup-container">
                <div class="help-btn" onclick="togglePopup('benchmark-help')">?</div>
                <div id="benchmark-help" class="popup-content">
                    <div class="popup-title">Comment utiliser</div>
                    <ul class="popup-list">
                        <li><b>Hauteur des barres</b> : Performance relative</li>
                        <li><b>Bleu</b> : Rendement moyen</li>
                        <li><b>Orange</b> : Volatilité (plus bas = mieux)</li>
                        <li><b>Action</b> : Ciblez les secteurs avec ratio rendement/risque favorable</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # KPI sectoriels
        sector_kpis = filtered_data.groupby('SECTEUR').agg({
            'SPREAD_POINTS': ['mean', 'std'],
            'DAYS_TO_MATURITY': 'median'
        })
        sector_kpis.columns = ['Rendement', 'Risque', 'Maturité']
        st.dataframe(
            sector_kpis.style.format("{:.1f}").background_gradient(cmap='Blues'),
            use_container_width=True
        )

        # =================================================================
        # SECTION 3 : RECOMMANDATIONS
        # =================================================================
        col_title, col_help = st.columns([0.9, 0.1])
        with col_title:
            st.subheader("🎯 Top Opportunités")
        with col_help:
            st.markdown("""
            <div class="popup-container">
                <div class="help-btn" onclick="togglePopup('reco-help')">?</div>
                <div id="reco-help" class="popup-content">
                    <div class="popup-title">Méthodologie</div>
                    <ul class="popup-list">
                        <li><b>Score</b> : 70% rendement + 30% maturité</li>
                        <li><b>Classement</b> : Percentile sectoriel</li>
                        <li><b>Utilisation</b> : Comparez les émetteurs d'un même secteur</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Calcul des meilleures opportunités
        top_emetteurs = filtered_data.groupby(['SECTEUR', 'PREFERREDNAMEISSUER']).agg({
            'SPREAD_POINTS': 'mean',
            'DAYS_TO_MATURITY': 'median'
        })
        top_emetteurs['Score'] = (
            top_emetteurs['SPREAD_POINTS'].rank(pct=True) * 0.7 + 
            (1 - top_emetteurs['DAYS_TO_MATURITY'].rank(pct=True)) * 0.3
        )
        
        # Affichage par secteur
        for secteur in top_emetteurs.index.get_level_values(0).unique():
            with st.expander(f"Secteur {secteur}", expanded=False):
                st.dataframe(
                    top_emetteurs.loc[secteur].nlargest(3, 'Score').style.format({
                        'SPREAD_POINTS': '{:.1f} pts',
                        'DAYS_TO_MATURITY': '{:.0f} jours',
                        'Score': '{:.2f}'
                    }),
                    use_container_width=True
                )

        # =================================================================
        # SECTION 4 : EVOLUTION TEMPORELLE
        # =================================================================
        col_title, col_help = st.columns([0.9, 0.1])
        with col_title:
            st.subheader("📈 Tendances")
        with col_help:
            st.markdown("""
            <div class="popup-container">
                <div class="help-btn" onclick="togglePopup('trend-help')">?</div>
                <div id="trend-help" class="popup-content">
                    <div class="popup-title">Analyse</div>
                    <ul class="popup-list">
                        <li><b>Pics</b> : Périodes de crise marché</li>
                        <li><b>Écarts</b> : Divergence sectorielle</li>
                        <li><b>Pente</b> : Amélioration/détérioration progressive</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Graphique temporel
        trend_data = filtered_data.groupby([
            filtered_data['ISSUEDT'].dt.to_period('M').astype(str),
            'SECTEUR'
        ])['SPREAD_POINTS'].mean().unstack()
        
        fig = px.line(
            trend_data,
            title="Évolution des spreads par secteur",
            labels={'value': 'Spread moyen (pts)', 'variable': 'Secteur'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("🔍 Exploration Avancée des Instruments")
        
        # =================================================================
        # SECTION 1 : MATRICE DE CORRELATION INTERACTIVE
        # =================================================================
        st.subheader("🧩 Relations entre Caractéristiques")
        
        # Préparation des données
        corr_data = filtered_data.copy()
        # Encodage des variables catégorielles
        cat_cols = ['INSTRCTGRY', 'INTERESTTYPE', 'INTERESTPERIODCTY', 'MATURITY_BUCKET']
        for col in cat_cols:
            corr_data[col] = corr_data[col].astype('category').cat.codes
        
        # Sélection des colonnes à inclure
        corr_cols = ['SPREAD_POINTS', 'DAYS_TO_MATURITY'] + cat_cols
        corr_matrix = corr_data[corr_cols].corr()
        
        # Visualisation interactive
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1,
            title="Corrélations entre caractéristiques",
            labels=dict(color="Coefficient"),
            x=corr_matrix.columns,
            y=corr_matrix.index,
            aspect="auto"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # =================================================================
        # SECTION 2 : EXPLORATION PAR FACETTES
        # =================================================================
        st.subheader("🔬 Analyse Multidimensionnelle")
        
        # Sélection des dimensions
        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis = st.selectbox("Axe X", ['SPREAD_POINTS', 'DAYS_TO_MATURITY'])
        with col2:
            y_axis = st.selectbox("Axe Y", ['INSTRCTGRY', 'INTERESTTYPE'])
        with col3:
            facet_col = st.selectbox("Facette", ['INTERESTPERIODCTY', 'MATURITY_BUCKET', 'None'])
        
        # Configuration conditionnelle
        facet_args = {'facet_col': facet_col} if facet_col != 'None' else {}
        
        # Visualisation dynamique
        fig = px.box(
            filtered_data,
            x=x_axis,
            y=y_axis,
            color='INSTRCTGRY',
            title=f"Distribution des {x_axis} par {y_axis}",
            **facet_args
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # =================================================================
        # SECTION 3 : CARTE DES INSTRUMENTS
        # =================================================================
        st.subheader("🗺️ Carte des Instruments")
        
        # Préparation des données
        instrument_map = filtered_data.groupby(['INSTRCTGRY', 'INTERESTTYPE']).agg({
            'SPREAD_POINTS': 'mean',
            'DAYS_TO_MATURITY': 'median',
            'ISSUEDT': 'count'
        }).reset_index()
        instrument_map.columns = ['Type', 'Intérêt', 'Spread Moyen', 'Maturité Médiane', 'Nombre']
        
        # Visualisation treemap interactive
        fig = px.treemap(
            instrument_map,
            path=['Type', 'Intérêt'],
            values='Nombre',
            color='Spread Moyen',
            color_continuous_scale='RdYlGn_r',
            hover_data=['Maturité Médiane'],
            title="Carte des instruments (taille = volume, couleur = spread)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # =================================================================
        # SECTION 4 : PROFILS TYPIQUES
        # =================================================================
        st.subheader("📊 Profils d'Instruments")
        
        # Cluster analysis simplifiée
        cluster_data = filtered_data.copy()
        # Encodage pour l'analyse
        for col in cat_cols:
            cluster_data[col] = cluster_data[col].astype('category').cat.codes
        
        # K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=4, random_state=42)
        features = ['SPREAD_POINTS', 'DAYS_TO_MATURITY'] + cat_cols
        cluster_data['Cluster'] = kmeans.fit_predict(cluster_data[features].fillna(0))
        
        # Visualisation des clusters
        fig = px.scatter(
            cluster_data,
            x='SPREAD_POINTS',
            y='DAYS_TO_MATURITY',
            color='Cluster',
            symbol='INSTRCTGRY',
            hover_name='INTERESTTYPE',
            title="Typologie des instruments (clustering automatique)",
            labels={
                'SPREAD_POINTS': 'Spread (points)',
                'DAYS_TO_MATURITY': 'Jours jusqu\'à maturité'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # =================================================================
        # SECTION 5 : COMPARAISON DES TOP INSTRUMENTS
        # =================================================================
        st.subheader("🏆 Top Performants par Catégorie")
        
        # Sélection de la métrique
        metric = st.radio(
            "Métrique de performance",
            ['SPREAD_POINTS', 'DAYS_TO_MATURITY'],
            horizontal=True
        )
        
        # Top 3 par catégorie
        categories = ['INSTRCTGRY', 'INTERESTTYPE', 'MATURITY_BUCKET']
        tabs = st.tabs([f"Top {cat}" for cat in categories])
        
        for tab, cat in zip(tabs, categories):
            with tab:
                top_data = filtered_data.groupby(cat)[metric].mean().nlargest(3)
                st.dataframe(
                    top_data.reset_index().rename(columns={cat: 'Catégorie', metric: 'Valeur'}),
                    hide_index=True,
                    use_container_width=True
                )
###########
    with tab5:
        # Section 2 - Affichage global des données filtrées
        st.header("Vue globale des émetteurs")
        st.dataframe(filtered_data, height=300)

        # Bouton de téléchargement des données globales
        st.download_button(
            label="Télécharger les données globales",
            data=filtered_data.to_csv(index=False).encode('utf-8'),
            file_name='donnees_globales.csv',
            mime='text/csv'
        )

        # Section 3 - Recherche et affichage par émetteur
        st.header("Recherche par émetteur")

        # Sélection de l'émetteur
        selected_issuer = st.selectbox(
            "Choisir un émetteur",
            options=sorted(filtered_data['PREFERREDNAMEISSUER'].unique()),
            index=0
        )

        # Filtrage des données pour l'émetteur sélectionné
        issuer_data = filtered_data[filtered_data['PREFERREDNAMEISSUER'] == selected_issuer]

        # Affichage sous forme de tableau
        if not issuer_data.empty:
            st.subheader(f"Détails pour {selected_issuer}")
            st.dataframe(issuer_data, height=300)
            
            # Bouton de téléchargement pour cet émetteur
            st.download_button(
                label=f"Télécharger les données de {selected_issuer}",
                data=issuer_data.to_csv(index=False).encode('utf-8'),
                file_name=f'donnees_{selected_issuer}.csv',
                mime='text/csv'
            )
        else:
            st.warning("Aucune donnée disponible pour cet émetteur avec les filtres actuels")
    ##########################33333        
    with tab6:
        st.header("🔮 Analyse et Prédiction des Spreads", divider="rainbow")
        
        # Vérification des données de base
        if 'processed_data' not in st.session_state:
            st.error("❌ Données non chargées. Veuillez compléter les étapes précédentes.")
            st.stop()
        
        # Utilisation des données filtrées ou complètes
        analysis_data = filtered_data if not filtered_data.empty else st.session_state.processed_data
        
        # Sélection de l'émetteur avec synchronisation multi-onglets
        selected_issuer = st.selectbox(
            "Sélectionner un émetteur",
            options=sorted(analysis_data['PREFERREDNAMEISSUER'].unique()),
            index=0,
            key="issuer_tab5"
        )
        
        # Filtrage des données
        issuer_data = analysis_data[analysis_data['PREFERREDNAMEISSUER'] == selected_issuer]
        
        if issuer_data.empty:
            st.warning(f"Aucune donnée disponible pour {selected_issuer}")
            st.stop()
        
        # Section Configuration
        with st.expander("⚙️ CONFIGURATION DU MODÈLE", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                model_choice = st.radio(
                    "Type de modèle",
                    ["Auto-sélection", "Choix manuel"],
                    horizontal=True
                )
                
                if model_choice == "Choix manuel":
                    model_type = st.selectbox(
                        "Modèle spécifique",
                        ["Lissage Exponentiel", "ARIMA", "Moyenne Mobile"],
                        index=0
                    )
            
            with col2:
                st.markdown("**Options avancées**")
                show_technical = st.checkbox("Afficher les diagnostics techniques")
                show_raw = st.checkbox("Afficher les données brutes")
        
        # Bouton d'exécution
        if st.button("▶️ Exécuter l'analyse", type="primary", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                try:
                    # Préparation des séries temporelles
                    ts_data = issuer_data.set_index('ISSUEDT').sort_index()['SPREAD_POINTS']
                    n_obs = len(ts_data)
                    
                    # Sélection automatique du modèle
                    if model_choice == "Auto-sélection":
                        if n_obs < 5:
                            model_type = "Moyenne Mobile"
                        elif n_obs < 12:
                            model_type = "Lissage Exponentiel"
                        else:
                            model_type = "ARIMA"
                    
                    # Entraînement des modèles
                    if model_type == "Moyenne Mobile":
                        model = ts_data.rolling(window=2, min_periods=1).mean()
                        forecast = model.iloc[-1]
                        conf_int = [forecast*0.9, forecast*1.1]
                        residuals = ts_data - model
                    
                    elif model_type == "Lissage Exponentiel":
                        model = ExponentialSmoothing(ts_data, trend='add').fit()
                        forecast = model.forecast(1).iloc[0]
                        conf_int = [forecast*0.95, forecast*1.05]
                        residuals = ts_data - model.fittedvalues
                    
                    elif model_type == "ARIMA":
                        model = ARIMA(ts_data, order=(1,1,1)).fit()
                        forecast_results = model.get_forecast(1)
                        forecast = forecast_results.predicted_mean.iloc[0]
                        conf_int = forecast_results.conf_int().iloc[0].tolist()
                        residuals = model.resid
                    
                    # Visualisation principale
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=ts_data.index, y=ts_data,
                        name='Données réelles',
                        line=dict(color='royalblue')
                    ))
                    
                    if hasattr(model, 'fittedvalues'):
                        fitted_vals = model.fittedvalues
                    else:
                        fitted_vals = model
                    
                    fig.add_trace(go.Scatter(
                        x=fitted_vals.index, y=fitted_vals,
                        name='Modèle ajusté',
                        line=dict(color='firebrick', dash='dash')
                    ))
                    
                    if model_type == "ARIMA":
                        fig.add_trace(go.Scatter(
                            x=fitted_vals.index,
                            y=fitted_vals + 1.96*residuals.std(),
                            fill=None, mode='lines',
                            line=dict(width=0),
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=fitted_vals.index,
                            y=fitted_vals - 1.96*residuals.std(),
                            fill='tonexty',
                            fillcolor='rgba(255,165,0,0.2)',
                            line=dict(width=0),
                            name='Intervalle de confiance'
                        ))
                    
                    fig.update_layout(
                        title=f"Évolution des spreads - {selected_issuer}",
                        xaxis_title='Date',
                        yaxis_title='Spread (points)',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Métriques de performance
                    st.subheader("📊 Résultats de la prédiction", divider="gray")
                    cols = st.columns(3)
                    last_val = ts_data.iloc[-1]
                    last_date = ts_data.index[-1].strftime('%Y-%m-%d')
                    delta = forecast - last_val
                    
                    cols[0].metric("Dernière valeur", 
                                f"{last_val:.1f} pts", 
                                f"Le {last_date}")
                    cols[1].metric("Prédiction", 
                                f"{forecast:.1f} pts", 
                                f"{delta:+.1f} pts ({delta/last_val*100:+.1f}%)",
                                delta_color="inverse")
                    cols[2].metric("Précision (MAE)", 
                                f"{mean_absolute_error(ts_data, fitted_vals):.1f} pts")
                    
                    # Diagnostics techniques
                    if show_technical:
                        st.subheader("🔍 Diagnostics techniques", divider="gray")
                        
                        # Résidus
                        fig_resid = px.histogram(
                            residuals, 
                            nbins=30,
                            title="Distribution des résidus",
                            labels={'value': 'Erreur (points)'}
                        )
                        st.plotly_chart(fig_resid, use_container_width=True)
                        
                        # ACF/PACF pour ARIMA
                        if model_type == "ARIMA":
                            st.markdown("**Analyse des autocorrélations**")
                            max_lags = min(10, (len(residuals)//2)-1)
                            
                            if max_lags > 0:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.pyplot(plot_acf(residuals, lags=max_lags))
                                with col2:
                                    st.pyplot(plot_pacf(residuals, lags=max_lags))
                            else:
                                st.warning("Trop peu de données pour l'analyse ACF/PACF")
                    
                    # Données brutes
                    if show_raw:
                        with st.expander("📁 Données brutes"):
                            st.dataframe(issuer_data.sort_values('ISSUEDT', ascending=False))
                            
                            csv = issuer_data.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="💾 Télécharger les données",
                                data=csv,
                                file_name=f"spreads_{selected_issuer}.csv",
                                mime='text/csv'
                            )
                    
                except Exception as e:
                    st.error(f"Erreur d'analyse: {str(e)}")
                    st.stop()
    with tab7:
        st.header("🎯 Modélisation du Spread - Comparaison de Modèles ML")

        # Vérification des données
        if 'processed_data' not in st.session_state:
            st.error("Veuillez d'abord charger et traiter les données dans les onglets précédents")
            st.stop()

        df = st.session_state.processed_data.copy()

        # Conversion des jours en années
        df['YEARS_TO_MATURITY'] = df['DAYS_TO_MATURITY'] / 365

        # Section Configuration
        with st.expander("⚙️ Configuration Commune des Modèles", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("% Données de test", 0.1, 0.5, 0.2)
                random_state = st.number_input("Random State", 42)
            with col2:
                # Sélection des modèles à entraîner
                selected_models = st.multiselect(
                    "Modèles à comparer",
                    ['Random Forest', 'Gradient Boosting', 'XGBoost'],
                    default=['Random Forest', 'Gradient Boosting']
                )

        # Configuration spécifique des modèles
        model_params = {}
        if 'Random Forest' in selected_models:
            with st.expander("🌳 Paramètres Random Forest"):
                col1, col2 = st.columns(2)
                with col1:
                    rf_n_estimators = st.slider("Nombre d'arbres (RF)", 50, 500, 100)
                    rf_max_depth = st.slider("Profondeur max (RF)", 3, 20, 10)
                model_params['Random Forest'] = {
                    'n_estimators': rf_n_estimators,
                    'max_depth': rf_max_depth
                }

        if 'Gradient Boosting' in selected_models:
            with st.expander("📈 Paramètres Gradient Boosting"):
                col1, col2 = st.columns(2)
                with col1:
                    gb_n_estimators = st.slider("Nombre d'arbres (GB)", 50, 500, 100)
                    gb_max_depth = st.slider("Profondeur max (GB)", 3, 10, 5)
                with col2:
                    gb_learning_rate = st.slider("Taux d'apprentissage (GB)", 0.01, 0.5, 0.1)
                model_params['Gradient Boosting'] = {
                    'n_estimators': gb_n_estimators,
                    'max_depth': gb_max_depth,
                    'learning_rate': gb_learning_rate
                }

        if 'XGBoost' in selected_models:
            with st.expander("🚀 Paramètres XGBoost"):
                col1, col2 = st.columns(2)
                with col1:
                    xgb_n_estimators = st.slider("Nombre d'arbres (XGB)", 50, 500, 100)
                    xgb_max_depth = st.slider("Profondeur max (XGB)", 3, 15, 6)
                with col2:
                    xgb_learning_rate = st.slider("Taux d'apprentissage (XGB)", 0.01, 0.5, 0.1)
                model_params['XGBoost'] = {
                    'n_estimators': xgb_n_estimators,
                    'max_depth': xgb_max_depth,
                    'learning_rate': xgb_learning_rate
                }

        # Préparation des données
        with st.spinner("Préparation des données..."):
            # Encodage des variables catégorielles
            cat_cols = ['PREFERREDNAMEISSUER', 'SECTEUR', 'INSTRCTGRY', 'INTERESTPERIODCTY', 'INTERESTTYPE', 'TYPETCN']
            df_encoded = pd.get_dummies(df, columns=cat_cols)
            
            # Sélection des features
            features = [col for col in df_encoded.columns if col.startswith(tuple(cat_cols)) or col in ['YEARS_TO_MATURITY']]
            X = df_encoded[features]
            y = df_encoded['SPREAD_POINTS']
            
            # Split train-test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        # Entraînement des modèles
        if st.button("🚀 Entraîner les Modèles", type="primary"):
            models = {}
            metrics = {}
            
            with st.spinner("Entraînement en cours..."):
                try:
                    # Random Forest
                    if 'Random Forest' in selected_models:
                        rf_model = RandomForestRegressor(
                            n_estimators=model_params['Random Forest']['n_estimators'],
                            max_depth=model_params['Random Forest']['max_depth'],
                            random_state=random_state
                        )
                        rf_model.fit(X_train, y_train)
                        y_pred = rf_model.predict(X_test)
                        models['Random Forest'] = rf_model
                        metrics['Random Forest'] = {
                            'MAE': mean_absolute_error(y_test, y_pred),
                            'R2': r2_score(y_test, y_pred)
                        }

                    # Gradient Boosting
                    if 'Gradient Boosting' in selected_models:
                        gb_model = GradientBoostingRegressor(
                            n_estimators=model_params['Gradient Boosting']['n_estimators'],
                            max_depth=model_params['Gradient Boosting']['max_depth'],
                            learning_rate=model_params['Gradient Boosting']['learning_rate'],
                            random_state=random_state
                        )
                        gb_model.fit(X_train, y_train)
                        y_pred = gb_model.predict(X_test)
                        models['Gradient Boosting'] = gb_model
                        metrics['Gradient Boosting'] = {
                            'MAE': mean_absolute_error(y_test, y_pred),
                            'R2': r2_score(y_test, y_pred)
                        }

                    # XGBoost
                    if 'XGBoost' in selected_models:
                        xgb_model = XGBRegressor(
                            n_estimators=model_params['XGBoost']['n_estimators'],
                            max_depth=model_params['XGBoost']['max_depth'],
                            learning_rate=model_params['XGBoost']['learning_rate'],
                            random_state=random_state
                        )
                        xgb_model.fit(X_train, y_train)
                        y_pred = xgb_model.predict(X_test)
                        models['XGBoost'] = xgb_model
                        metrics['XGBoost'] = {
                            'MAE': mean_absolute_error(y_test, y_pred),
                            'R2': r2_score(y_test, y_pred)
                        }

                    # Stockage en session
                    st.session_state.models = models
                    st.session_state.metrics = metrics
                    st.session_state.rf_features = features
                    
                    # Préparation des mappings hiérarchiques
                    st.session_state.secteur_to_emetteurs = df.groupby('SECTEUR')['PREFERREDNAMEISSUER'].unique().to_dict()
                    st.session_state.all_secteurs = df['SECTEUR'].unique()
                    st.session_state.all_instruments = df['INSTRCTGRY'].unique()
                    st.session_state.all_interest_periods = df['INTERESTPERIODCTY'].unique()
                    st.session_state.all_interest_types = df['INTERESTTYPE'].unique()
                    st.session_state.all_tcn_types = df['TYPETCN'].unique()
                    
                    st.success("Modèles entraînés avec succès!")
                except Exception as e:
                    st.error(f"Erreur lors de l'entraînement: {str(e)}")
                    st.stop()

        # Affichage des résultats si disponible
        if 'models' in st.session_state:
            # Comparaison des modèles
            st.subheader("🏆 Comparaison des Performances")
            
            # Création d'un DataFrame pour les métriques
            metrics_df = pd.DataFrame.from_dict(st.session_state.metrics, orient='index')
            metrics_df.reset_index(inplace=True)
            metrics_df.rename(columns={'index': 'Modèle'}, inplace=True)
            
            # Affichage des métriques
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Meilleur MAE", f"{metrics_df['MAE'].min():.2f}")
            with col2:
                st.metric("Meilleur R²", f"{metrics_df['R2'].max():.2f}")
            
            # Graphique de comparaison
            fig_comp = px.bar(
                metrics_df.melt(id_vars='Modèle', var_name='Metric', value_name='Value'),
                x='Modèle',
                y='Value',
                color='Metric',
                barmode='group',
                title='Comparaison des Métriques par Modèle'
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Sélection du modèle optimal
            st.subheader("🏅 Sélection du Modèle Optimal")
            best_model = st.selectbox(
                "Choisissez le modèle à utiliser pour le scoring",
                options=list(st.session_state.models.keys()),
                index=metrics_df['R2'].idxmax()
            )
            
            # Stockage du modèle sélectionné pour le scoring
            st.session_state.selected_model = best_model
            st.session_state.selected_model_obj = st.session_state.models[best_model]
            
            st.success(f"Modèle {best_model} sélectionné pour le scoring dans l'onglet suivant!")
            
            # Feature Importance du modèle sélectionné
            st.subheader(f"🔍 Importance des Variables ({best_model})")
            
            if best_model == 'Random Forest':
                importances = st.session_state.models[best_model].feature_importances_
            elif best_model == 'Gradient Boosting':
                importances = st.session_state.models[best_model].feature_importances_
            else:  # XGBoost
                importances = st.session_state.models[best_model].feature_importances_
            
            feature_imp = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig_imp = px.bar(
                feature_imp.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title=f'Top 10 des variables les plus importantes ({best_model})'
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
            # Section de prédiction interactive
            st.subheader("🔎 Prédiction Interactive")
            
            with st.form("predict_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sélection hiérarchique Secteur -> Émetteur
                    secteur = st.selectbox("Secteur", st.session_state.all_secteurs)
                    emetteurs_disponibles = st.session_state.secteur_to_emetteurs.get(secteur, [])
                    emetteur = st.selectbox("Émetteur", emetteurs_disponibles)
                    
                    instrument_cat = st.selectbox("Catégorie d'instrument", st.session_state.all_instruments)
                    
                    if instrument_cat == 'TCN':
                        tcn_type = st.selectbox("Type de TCN", st.session_state.all_tcn_types)
                    
                with col2:
                    interest_period = st.selectbox("Période d'intérêt", st.session_state.all_interest_periods)
                    interest_type = st.selectbox("Type d'intérêt", st.session_state.all_interest_types)
                    years_to_maturity = st.number_input("Années jusqu'à maturité", 
                                                      min_value=0.1, 
                                                      value=1.0, 
                                                      step=0.1, 
                                                      format="%.1f")
                
                submitted = st.form_submit_button("Prédire le Spread")
                
                if submitted:
                    try:
                        # Préparation des données pour la prédiction
                        input_data = {
                            'YEARS_TO_MATURITY': years_to_maturity,
                            'PREFERREDNAMEISSUER_' + emetteur: 1,
                            'SECTEUR_' + secteur: 1,
                            'INSTRCTGRY_' + instrument_cat: 1,
                            'INTERESTPERIODCTY_' + interest_period: 1,
                            'INTERESTTYPE_' + interest_type: 1
                        }
                        
                        if instrument_cat == 'TCN':
                            input_data['TYPETCN_' + tcn_type] = 1
                        
                        pred_df = pd.DataFrame(0, index=[0], columns=st.session_state.rf_features)
                        
                        for key, value in input_data.items():
                            if key in pred_df.columns:
                                pred_df[key] = value
                        
                        predicted_spread = st.session_state.models[best_model].predict(pred_df)[0]
                        
                        st.success(f"Spread prédit ({best_model}) pour {emetteur}: {predicted_spread:.2f} points")
                        
                    except Exception as e:
                        st.error(f"Erreur lors de la prédiction: {str(e)}")
            
            # Visualisation par Émetteur - Réel vs Prédit
            st.subheader("📈 Comparaison des Spreads Réels vs Prédits")
            
            # Sélection de l'émetteur
            secteur_viz = st.selectbox("Sélectionnez un secteur", st.session_state.all_secteurs, key='secteur_viz')
            emetteurs_disponibles_viz = st.session_state.secteur_to_emetteurs.get(secteur_viz, [])
            selected_emetteur_viz = st.selectbox("Sélectionner un émetteur", emetteurs_disponibles_viz, key='emetteur_viz')

            if selected_emetteur_viz:
                emetteur_data = df[df['PREFERREDNAMEISSUER'] == selected_emetteur_viz].copy()
                
                if not emetteur_data.empty:
                    with st.spinner("Calcul des prédictions..."):
                        # Préparation des données pour la prédiction
                        emetteur_encoded = pd.get_dummies(emetteur_data, columns=cat_cols)
                        
                        # Ajout des colonnes manquantes avec valeur 0
                        missing_cols = set(st.session_state.rf_features) - set(emetteur_encoded.columns)
                        for col in missing_cols:
                            emetteur_encoded[col] = 0
                        
                        X_emetteur = emetteur_encoded[st.session_state.rf_features]
                        
                        # Prédiction avec le modèle sélectionné
                        emetteur_data['PREDICTED_SPREAD'] = st.session_state.selected_model_obj.predict(X_emetteur)
                        
                        # Trier par date pour une bonne visualisation
                        emetteur_data = emetteur_data.sort_values('ISSUEDT')
                        
                        # Création du graphique
                        fig = go.Figure()
                        
                        # Spread réel (bleu)
                        fig.add_trace(go.Scatter(
                            x=emetteur_data['ISSUEDT'],
                            y=emetteur_data['SPREAD_POINTS'],
                            name='Spread Réel',
                            mode='markers+lines',
                            marker=dict(color='blue', size=8),
                            line=dict(color='blue', width=1)
                        ))
                        
                        # Spread prédit (rouge)
                        fig.add_trace(go.Scatter(
                            x=emetteur_data['ISSUEDT'],
                            y=emetteur_data['PREDICTED_SPREAD'],
                            name=f'Spread Prédit ({best_model})',
                            mode='markers+lines',
                            marker=dict(color='red', size=8),
                            line=dict(color='red', width=1)
                        ))
                        
                        # Mise en forme du graphique
                        fig.update_layout(
                            title=f'Comparaison des Spreads pour {selected_emetteur_viz}',
                            xaxis_title='Date d\'émission',
                            yaxis_title='Spread (points)',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calcul et affichage des erreurs
                        emetteur_data['ERREUR'] = emetteur_data['PREDICTED_SPREAD'] - emetteur_data['SPREAD_POINTS']
                        emetteur_data['ERREUR_ABS'] = abs(emetteur_data['ERREUR'])
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("MAE moyen", f"{emetteur_data['ERREUR_ABS'].mean():.2f} points")
                        col2.metric("Erreur max", f"{emetteur_data['ERREUR_ABS'].max():.2f} points")
                        col3.metric("Précision moyenne", f"{1 - (emetteur_data['ERREUR_ABS'].mean()/emetteur_data['SPREAD_POINTS'].mean()):.2%}")
                        
                        # Affichage des données sous forme de tableau
                        st.subheader("📊 Données Détailées")
                        st.dataframe(emetteur_data[[
                            'ISSUEDT', 
                            'SPREAD_POINTS', 
                            'PREDICTED_SPREAD', 
                            'ERREUR',
                            'YEARS_TO_MATURITY',
                            'SECTEUR',
                            'INSTRCTGRY'
                        ]].sort_values('ISSUEDT', ascending=False))
                        
                        # Téléchargement des résultats pour l'émetteur sélectionné
                        csv = emetteur_data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Télécharger les prédictions pour l'émetteur",
                            data=csv,
                            file_name=f"spreads_{selected_emetteur_viz}.csv",
                            mime="text/csv"
                        )
                        
                        # Téléchargement des prédictions pour tous les émetteurs
                        with st.spinner("Génération des prédictions pour tous les émetteurs..."):
                            # Préparation des données pour tous les émetteurs
                            all_encoded = pd.get_dummies(df, columns=cat_cols)
                            
                            # Ajout des colonnes manquantes avec valeur 0
                            missing_cols = set(st.session_state.rf_features) - set(all_encoded.columns)
                            for col in missing_cols:
                                all_encoded[col] = 0
                            
                            X_all = all_encoded[st.session_state.rf_features]
                            
                            # Prédiction avec le modèle sélectionné
                            df['PREDICTED_SPREAD'] = st.session_state.selected_model_obj.predict(X_all)
                            
                            # Calcul des erreurs
                            df['ERREUR'] = df['PREDICTED_SPREAD'] - df['SPREAD_POINTS']
                            df['ERREUR_ABS'] = abs(df['ERREUR'])
                            
                            # Sélection des colonnes pour l'export
                            export_columns = [
                                'ISSUEDT', 
                                'PREFERREDNAMEISSUER',
                                'SPREAD_POINTS', 
                                'PREDICTED_SPREAD', 
                                'ERREUR',
                                'YEARS_TO_MATURITY',
                                'SECTEUR',
                                'INSTRCTGRY',
                                'INTERESTPERIODCTY',
                                'INTERESTTYPE'
                            ]
                            if 'TYPETCN' in df.columns:
                                export_columns.append('TYPETCN')
                            
                            all_predictions = df[export_columns].sort_values(['PREFERREDNAMEISSUER', 'ISSUEDT'])
                            
                            # Téléchargement du fichier CSV pour tous les émetteurs
                            all_csv = all_predictions.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Télécharger les prédictions pour tous les émetteurs",
                                data=all_csv,
                                file_name="spreads_all_emetteurs.csv",
                                mime="text/csv"
                            )
                else:
                    st.warning(f"Aucune donnée disponible pour {selected_emetteur_viz}")
        else:
            st.info("Veuillez entraîner les modèles pour voir les résultats et faire des prédictions")

            
    with tab8:
        st.header("📈 Scoring des Émetteurs par Modèle de Spread")
        
        # Fonctions utilitaires
        def get_rating_color(rating):
            """Retourne la couleur associée à une notation"""
            colors = {
                'AAA': '#006400', 'AA+': '#008000', 'AA': '#38A800',
                'A+': '#90EE90', 'A': '#FFD700', 'A-': '#FFA500',
                'BBB+': '#FF8C00', 'BBB': '#FF4500', 'BBB-': '#FF0000',
                'BB+': '#CD5C5C', 'BB': '#B22222', 'BB-': '#8B0000'
            }
            return colors.get(rating, '#FFFFFF')

        # Vérification des prérequis
        if 'processed_data' not in st.session_state or 'selected_model_obj' not in st.session_state:
            st.error("Veuillez d'abord :\n1. Charger les données\n2. Entraîner un modèle dans l'onglet Modélisation")
            st.stop()
        
        # Récupération des données
        df = st.session_state.processed_data.copy()
        model = st.session_state.selected_model_obj
        
        # Section de configuration
        with st.expander("⚙️ Paramètres du Scoring", expanded=True):
            st.markdown("""
            **Méthodologie** :
            1. Utilisation du modèle pré-entraîné pour prédire le spread théorique
            2. Comparaison du spread réel avec le spread prédit
            3. Classement des émetteurs selon l'écart type
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                min_obs = st.number_input("Nombre minimum d'émissions", 
                                        min_value=1, max_value=50, value=5)
                maturity_filter = st.slider("Filtrer par maturité (années)", 
                                        1.0, 30.0, (1.0, 10.0))
            with col2:
                risk_metric = st.selectbox("Métrique de risque", 
                                        ['Spread Absolu', 'Spread Relatif', 'Z-Score'])
                confidence_level = st.slider("Niveau de confiance", 80, 99, 90)

        # Filtrage des émetteurs avec suffisamment d'observations
        issuer_counts = df['PREFERREDNAMEISSUER'].value_counts()
        valid_issuers = issuer_counts[issuer_counts >= min_obs].index.tolist()
        df_filtered = df[df['PREFERREDNAMEISSUER'].isin(valid_issuers)]
        
        # Filtrage par maturité
        df_filtered = df_filtered[
            (df_filtered['YEARS_TO_MATURITY'] >= maturity_filter[0]) & 
            (df_filtered['YEARS_TO_MATURITY'] <= maturity_filter[1])
        ]

        if len(df_filtered) == 0:
            st.warning("Aucun émetteur ne correspond aux critères sélectionnés")
            st.stop()

        # Calcul des scores
        if st.button("🏁 Calculer les Scores"):
            with st.spinner("Calcul des scores en cours..."):
                try:
                    # Préparation des données pour la prédiction
                    X = df_filtered[st.session_state.rf_features]
                    
                    # Prédiction du spread théorique
                    df_filtered['SPREAD_PRED'] = model.predict(X)
                    
                    # Calcul des métriques par émetteur
                    issuer_stats = df_filtered.groupby('PREFERREDNAMEISSUER').agg({
                        'SPREAD_POINTS': ['mean', 'std', 'count'],
                        'SPREAD_PRED': 'mean',
                        'SECTEUR': 'first'
                    })
                    
                    issuer_stats.columns = ['SPREAD_MEAN', 'SPREAD_STD', 'OBS_COUNT', 'SPREAD_PRED', 'SECTEUR']
                    
                    # Calcul des scores
                    issuer_stats['SPREAD_DIFF'] = issuer_stats['SPREAD_MEAN'] - issuer_stats['SPREAD_PRED']
                    issuer_stats['Z_SCORE'] = (issuer_stats['SPREAD_MEAN'] - issuer_stats['SPREAD_MEAN'].mean()) / issuer_stats['SPREAD_MEAN'].std()
                    
                    # Classement selon la métrique choisie
                    if risk_metric == 'Spread Absolu':
                        issuer_stats['SCORE'] = issuer_stats['SPREAD_MEAN']
                    elif risk_metric == 'Spread Relatif':
                        issuer_stats['SCORE'] = issuer_stats['SPREAD_DIFF']
                    else:
                        issuer_stats['SCORE'] = issuer_stats['Z_SCORE']
                    
                    # Normalisation des scores (0-100)
                    issuer_stats['SCORE_NORM'] = 100 - ((issuer_stats['SCORE'] - issuer_stats['SCORE'].min()) / 
                                                    (issuer_stats['SCORE'].max() - issuer_stats['SCORE'].min()) * 40 + 60)
                    
                    # Notation
                    def map_to_rating(score):
                        if score >= 90: return 'AAA'
                        elif score >= 85: return 'AA+'
                        elif score >= 80: return 'AA'
                        elif score >= 75: return 'AA-'
                        elif score >= 70: return 'A+'
                        elif score >= 65: return 'A'
                        elif score >= 60: return 'A-'
                        elif score >= 55: return 'BBB+'
                        elif score >= 50: return 'BBB'
                        elif score >= 45: return 'BBB-'
                        elif score >= 40: return 'BB+'
                        elif score >= 35: return 'BB'
                        else: return 'BB-'
                    
                    issuer_stats['NOTATION'] = issuer_stats['SCORE_NORM'].apply(map_to_rating)
                    
                    # Sauvegarde des résultats
                    st.session_state.issuer_scores = issuer_stats.sort_values('SCORE_NORM', ascending=False)
                    
                except Exception as e:
                    st.error(f"Erreur lors du calcul : {str(e)}")
                    st.stop()

        # Affichage des résultats
        if 'issuer_scores' in st.session_state:
            issuer_scores = st.session_state.issuer_scores
            
            st.success(f"Scoring terminé pour {len(issuer_scores)} émetteurs")
            
            # KPI
            col1, col2, col3 = st.columns(3)
            col1.metric("Meilleur Score", 
                      f"{issuer_scores['SCORE_NORM'].max():.1f}", 
                      issuer_scores['NOTATION'].iloc[0])
            col2.metric("Score Moyen", 
                      f"{issuer_scores['SCORE_NORM'].mean():.1f}")
            col3.metric("Pire Score", 
                      f"{issuer_scores['SCORE_NORM'].min():.1f}", 
                      issuer_scores['NOTATION'].iloc[-1])
            
            # Tableau des résultats
            st.subheader("🏆 Classement des Émetteurs")
            
            # Fonction de style
            def color_row(row):
                color = get_rating_color(row['NOTATION'])
                return ['background-color: ' + color] * len(row)
            
            # Affichage du tableau
            display_cols = ['SECTEUR', 'OBS_COUNT', 'SPREAD_MEAN', 'SPREAD_PRED', 
                          'SPREAD_DIFF', 'SCORE_NORM', 'NOTATION']
            
            st.dataframe(
                issuer_scores[display_cols]
                .rename(columns={
                    'SECTEUR': 'Secteur',
                    'OBS_COUNT': 'Nb Émissions',
                    'SPREAD_MEAN': 'Spread Moyen',
                    'SPREAD_PRED': 'Spread Prédit',
                    'SPREAD_DIFF': 'Écart',
                    'SCORE_NORM': 'Score',
                    'NOTATION': 'Notation'
                })
                .style.apply(color_row, axis=1),
                height=600
            )
            
            # Visualisation
            st.subheader("📊 Analyse Sectorielle")
            
            fig = px.box(
                issuer_scores.reset_index(),
                x='SECTEUR',
                y='SCORE_NORM',
                color='SECTEUR',
                points="all",
                hover_data=['PREFERREDNAMEISSUER'],
                labels={'SCORE_NORM': 'Score', 'SECTEUR': 'Secteur'},
                title="Distribution des Scores par Secteur"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Export
            csv = issuer_scores.reset_index().to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Exporter les résultats complets",
                data=csv,
                file_name="scores_emetteurs.csv",
                mime="text/csv"
            )   
    # Statistiques globales
    st.sidebar.header("Statistiques Globales")
    st.sidebar.metric("Nombre d'émetteurs", filtered_data['PREFERREDNAMEISSUER'].nunique())
    st.sidebar.metric("Nombre d'instruments", filtered_data.shape[0])
    st.sidebar.metric("Spread moyen (points)", f"{filtered_data['SPREAD_POINTS'].mean():.1f}" if not filtered_data.empty else "N/A")
    st.sidebar.metric("Spread médian (points)", f"{filtered_data['SPREAD_POINTS'].median():.1f}" if not filtered_data.empty else "N/A")

else:

    st.info("Veuillez compléter toutes les étapes de traitement des données pour accéder à l'analyse.")   


