const decorateWIndex = (v: number, i: number) => [v, i];
const removeDecoration = (a: Array<any>) => a[1]; // leave only index

export function argsort(
  arr: Array<number>,
  ascending?: boolean,
): Array<number> {
  const sortAscending = ascending === undefined ? true : ascending;
  let sorted = arr
    .map(decorateWIndex)
    .sort((a, b) => a[0] - b[0])
    .map(removeDecoration);
  if (sortAscending === true) {
    return sorted;
  } else {
    return sorted.reverse();
  }
}
