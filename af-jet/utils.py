
def parse_indices(index_str):
  str_indexes = index_str.split(',')
  indices = []
  for index in str_indexes:
    index = index.strip()
    if ":" in index:
      range_split = index.split(':')
      if len(range_split)==2:
        start = int(range_split[0])
        end = int(range_split[1])
        for val in range(start,end+1):
          indices.append(val)
      elif len(range_split)==3:
        start = int(range_split[0])
        end = int(range_split[1])
        step = int(range_split[2])
        for val in range(start,end+1,step):
          indices.append(val)
      else:
        print("Error: invalid range call")
    else:
      val = index.strip()
      indices.append(int(val))
  return indices

