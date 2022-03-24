const decorateWIndex = (v: number, i: number) => [v, i];
const removeDecoration = (a: Array<any>) => a[1]; // leave only index

export function argSort(
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

export function cumSum(arr: Array<number>): Array<number> {
  const cumulativeSum = (
    (sum: number) => (value: number) =>
      (sum += value)
  )(0);
  return arr.map(cumulativeSum);
}

export function repeat(arr: Array<any>, times: Array<number>): Array<any> {
  return arr.flatMap((e, index) => Array(times[index]).fill(e));
}
