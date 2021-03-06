import spacy

from wellcomeml.ml import SpacyNER
from wellcomeml.metrics.ner_classification_report import ner_classification_report

X_train = [
        'n Journal of Psychiatry 158: 2071–4\nFreeman MP, Hibbeln JR, Wisner KL et al. (2006)\nOmega-3 fatty ac',
        'rd, (BKKBN)\n \nJakarta, Indonesia\n29. Drs Titut Prihyugiarto\n MSPA\n \nSenior Researcher for Reproducti',
        'a Santé, 2008. \n118. Konradsen, F. et coll. Community uptake of safe storage boxes to reduce self-po',
        'ted that the two treatments can \nbe combined. Contrarily, Wapf et al. \nnoted that many treatment per',
        'ti-tuberculosis treatment in Mongolia. Int J Tuberc Lung Dis. 2015;19(6):657–62. \n160. Dudley L, Aze',
        'he \nScottish Heart Health Study: cohort study. BMJ, 1997, 315:722–729. \nUmesawa M, Iso H, Date C et ',
        'T.A., G. Marland, and R.J. Andres (2010). Global, Regional, and National Fossil-Fuel CO2 Emissions. ',
        'Ian Gr\nMr Ian Graayy\nPrincipal Policy Officer (Public Health and Health Protection), Chartered Insti',
        '. \n3. \nFischer G and Stöver H. Assessing the current state of opioid-dependence treatment across Eur',
        'ated by\nLlorca et al. (2014) or Pae et al. (2015), or when vortioxetine was assumed to be\nas effecti',
]

y_train = [
        [{'start': 36, 'end': 46, 'label': 'PERSON'}, {'start': 48, 'end': 58, 'label': 'PERSON'}, {'start': 61, 'end': 69, 'label': 'PERSON'}],
        [{'start': 41, 'end': 59, 'label': 'PERSON'}],
        [{'start': 21, 'end': 34, 'label': 'PERSON'}],
        [{'start': 58, 'end': 62, 'label': 'PERSON'}],
        [{'start': 87, 'end': 95, 'label': 'PERSON'}],
        [{'start': 72, 'end': 81, 'label': 'PERSON'}, {'start': 83, 'end': 88, 'label': 'PERSON'}, {'start': 90, 'end': 96, 'label': 'PERSON'}],
        [{'start': 6, 'end': 16, 'label': 'PERSON'}, {'start': 22, 'end': 33, 'label': 'PERSON'}],
        [{'start': 0, 'end': 6, 'label': 'PERSON'}, {'start': 10, 'end': 20, 'label': 'PERSON'}],
        [{'start': 7, 'end': 16, 'label': 'PERSON'}, {'start': 21, 'end': 30, 'label': 'PERSON'}],
        [{'start': 8, 'end': 14, 'label': 'PERSON'}, {'start': 32, 'end': 35, 'label': 'PERSON'}],
]

# A list of the groups each of the data points belong to
groups = ['Group 1', 'Group 2', 'Group 3', 'Group 2', 'Group 1', 'Group 3', 'Group 3', 'Group 3', 'Group 2', 'Group 1']

spacy_ner = SpacyNER(n_iter=3, dropout=0.2, output=True)
spacy_ner.load("en_core_web_sm")
nlp = spacy_ner.fit(X_train, y_train)

# Predict the entities in a piece of text
text = '\nKhumalo, Lungile, National Department of Health \n• \nKistnasamy, Dr Barry, National Department of He'
predictions = spacy_ner.predict(text)
print([text[entity['start']: entity['end']] for entity in predictions])

# Evaluate the performance of the model on the training data
y_pred = [spacy_ner.predict(text) for text in X_train]
f1 = spacy_ner.score(y_train, y_pred, tags=['PERSON'])
print(f1)

# Evaluate the performance of the model per group
report = ner_classification_report(y_train, y_pred, groups, tags=['PERSON'])
print(report)

