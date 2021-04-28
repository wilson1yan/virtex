import sys
import json

files = sys.argv[1:]
print(f'Found {len(files)} files to merge:', files)

output = dict()
for f in files:
    f_data = json.load(open(f, 'r'))
    output.update(f_data)
print(f"Final output size: {len(output)}")

json.dump(output, open('outputs/captions.json', 'w'))

