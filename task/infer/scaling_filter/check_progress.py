import re

path = '/opt/tiger/script/20250922.log'
lines = []
with open(path, 'r', encoding='utf8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        lines.append(line)

examples = []
for line in lines:
    result = re.match('.*part-(\d+)-.*', line)
    if not result:
        continue
    example = result.group(1)
    example = int(example)
    examples.append(example)

left_ids = []
max_id_plus_one = 2000
for i in range(max_id_plus_one):
    if i not in examples:
        left_ids.append(i)
print(left_ids)
print(len(left_ids))