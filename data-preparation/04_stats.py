"""Per-category multilabel response counts; no inference or API calls."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from common import LABELS, config, connect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--csv', help='Optional UTF-8 CSV output')
    args = parser.parse_args()
    db = connect(config(args.config)['db'])
    query = 'SELECT p.payload,r.label FROM responses r JOIN prompts p ON p.id=r.prompt_id'
    groups = defaultdict(lambda: {'positive': Counter(), 'total': 0, 'safe': 0, 'pending': 0, 'uncertain': 0, 'unusable': 0})
    for row in db.execute(query):
        modality = '图文' if json.loads(row['payload']).get('images') else '文本'
        group = groups[modality]
        if not row['label']:
            group['pending'] += 1
            continue
        label = json.loads(row['label'])
        if not label.get('usable', True):
            group['unusable'] += 1
            continue
        if label['safety'] == 'uncertain':
            group['uncertain'] += 1
            continue
        group['total'] += 1
        group['safe'] += label['safety'] == 'safe'
        group['positive'].update(label['categories'])
    output = []
    for modality, group in sorted(groups.items()):
        print(f'\n{modality} | 有效 {group["total"]:,} | 安全 {group["safe"]:,} | 待标 {group["pending"]:,} | 待判 {group["uncertain"]:,} | 无效 {group["unusable"]:,}')
        print(f'{"ID":<5} {"类别":<18} {"正例":>9} {"负例":>9} {"正例比例":>9}')
        print('─' * 78)
        for category in sorted(LABELS, key=lambda c: (-group['positive'][c], c)):
            positive = group['positive'][category]
            negative = group['total'] - positive
            ratio = positive / group['total'] if group['total'] else 0
            name = LABELS[category]
            padding = ' ' * max(1, 28 - sum(2 if ord(c) > 127 else 1 for c in name))
            print(f'{category:<5} {name}{padding}{positive:>9,} {negative:>9,} {ratio:>9.1%}')
            output.append([modality, category, name, positive, negative, group['safe'], ratio])
    print('\n负例＝已明确标注且不含本类风险的回答（包括安全回答和其他风险回答）。多标签正例不可相加当作总样本数。')
    print('任务状态:', dict(db.execute('SELECT state,count(*) FROM prompts GROUP BY state').fetchall()))
    if args.csv:
        with open(args.csv, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['modality', 'category', 'name', 'positive', 'negative', 'safe', 'positive_ratio'])
            writer.writerows(output)
    db.close()


if __name__ == '__main__':
    main()
