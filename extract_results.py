import json
with open('Markovian_analysis.ipynb') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'stream' and out.get('name') == 'stdout':
                text = ''.join(out['text'])
                if any(k in text for k in ['CK', 'Predictability', 'Timescale', 'VAMP', 'Implied']):
                    print(text)
