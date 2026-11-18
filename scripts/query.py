#!/usr/bin/env python3
"""Read values out of the released data files and print one number, or a plain-text table.

SPEC: a value inside a data file, written as the file path followed by the keys or list indices to follow,
joined by '::', e.g. data/results/x.json::labeled::rf::f1.
  - .json files are read whole; .jsonl files are the list of their records (one per line); .csv files are the
    list of their rows (a dict per row, all values text).
  - '*' expands to every key of a dict or every item of a list, so one SPEC can name many values. A key with
    glob characters ('flip_*', 'labeled|*', 'linear_s?d') expands to the matching keys of a dict. A key that
    exists literally is always taken literally.
  - Null values are skipped. --skip-key KEY leaves out dict entries named KEY when a pattern expands a dict.
  - In a table, a row is labeled by the keys the patterns matched, joined by '/'.
RECORDS: a SPEC naming records (dicts); a SPEC naming a list (a CSV or JSON-lines file, a list in a JSON file)
counts as its items. FIELD: a key inside a record, or a '::' path of keys (lin_exact::n_target).
STAT: min, max, median, mean, sum, count (number of values), max-abs (largest absolute value).

Values
  value SPEC                          the one value at SPEC
  stat STAT SPEC [SPEC ...]           STAT over every value the SPECs name
  above-one STAT SPEC [SPEC ...]      STAT over each value minus 1
  nth-largest N SPEC [SPEC ...]       the N-th largest value
  largest-share N SPEC [SPEC ...]     sum of the N largest values over the sum of all values
  weighted-mean VALUES WEIGHTS        mean of the values VALUES names, weighted by those WEIGHTS names (in order)
  one-over-count SPEC                 1 divided by the number of values SPEC names
  log {2,10,e} SPEC                   logarithm of the value at SPEC
Pairs of values
  diff A B                            A minus B (one value each)
  ratio A B                           A divided by B (one value each)
  diff-stat STAT A B [A B ...]        STAT over the differences A - B, pair by pair
  ratio-stat STAT A B [A B ...]       STAT over the ratios A / B, pair by pair
                                      within a pair, A and B name equally many values (paired in file order) or
                                      one of them names a single value; --best first reduces each SPEC to the
                                      largest value it names (the better of several models)
  count-equal A B                     positions at which the values A names equal those B names (in file order)
Records
  count RECORDS [RECORDS ...]         records passing the filters; --distinct FIELD: their distinct FIELD values
  share RECORDS --share-where COL=VAL | --share-where-in COL=V1,V2 | --share-where-not COL=VAL
                                      share of the filtered records that meet the share condition
  field-stat STAT RECORDS [RECORDS ...] --field FIELD
                                      STAT over the non-null FIELD values of the filtered records
  group-size RECORDS [RECORDS ...] --by FIELD [--distinct FIELD]
                                      size of each group of records sharing FIELD (groups are formed within each
                                      RECORDS argument); with --distinct, a group's number of distinct values
  group-stat STAT RECORDS --by FIELD --field FIELD
                                      STAT of the non-null FIELD values within each group
  share ... --by FIELD                the share within each group
  Without a reduction, group-size, group-stat and share --by print one row per group. Reductions:
    --over-groups STAT                STAT over the groups' values
    --exclude-largest                 first leave out the groups whose value is the largest
    --count-groups-above X            number of groups whose value is above X
    --count-groups-at-max             number of groups whose value equals the largest
    --largest-share N                 (group-size) records in the N largest groups over all records
    --largest-group-where COL=VAL     (group-size) records of the largest group (the first in file order) that
                                      have COL equal to VAL
  Record filters (repeatable, all must hold):
    --where FIELD=VAL     FIELD equals VAL          --where-in FIELD=V1,V2,...  FIELD equals one of the values
    --where-not FIELD=VAL FIELD differs from VAL    --nonempty FIELD            FIELD is set (not null, not empty)
    --above FIELD=X       FIELD is a number above X --below FIELD=X             FIELD is a number below X
Tables
  table SPEC [--field FIELD ...]      one row per value SPEC names, one column per FIELD; a '*' in FIELD gives one
                                      column per key; without --field, one column holding the value
  grid-table GRID SPEC [SPEC ...]     one row per point of the list at GRID, one column per list the SPECs name
  value-at-grid-point SPEC --grid GRID --at X
                                      the item of the list at SPEC where the list at GRID equals X

Output format (precision only): --digits N rounds to N decimals (negative N: to tens, hundreds, ...);
--rounding half-up (default), half-even or ceiling; --percent multiplies by 100, --permille by 1000 and
--in-millions divides by 1,000,000 first; --sep adds thousands separators; --significant N prints N
significant digits and --sci N prints N decimals in scientific notation. Without these, whole numbers print
without decimals and other numbers in full. In a table, a missing value prints as '--'.
"""
import argparse
import csv
import fnmatch
import json
import math
import statistics
import sys
from decimal import ROUND_CEILING, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal

STATS = ["min", "max", "median", "mean", "sum", "count", "max-abs"]


def resolve(spec, skip_keys):
    """Every non-null (label, value) a SPEC names, in file order; the label joins the keys patterns matched."""
    parts = spec.split("::")
    # the root value of the file: JSON lines and CSV files are the list of their records
    path = parts[0]
    if path.endswith(".jsonl"):
        root = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    root.append(json.loads(line))
    elif path.endswith(".csv"):
        with open(path, newline="") as f:
            root = list(csv.DictReader(f))
    else:
        with open(path) as f:
            root = json.load(f)
    current = [("", root)]
    for key in parts[1:]:
        following = []
        for label, node in current:
            if isinstance(node, dict):
                if key in node:
                    following.append((label, node[key]))
                    continue
                if "*" not in key and "?" not in key and "[" not in key:
                    sys.exit(f"query.py: key {key!r} not found in {spec}")
                for k in node:
                    if k in skip_keys or not fnmatch.fnmatchcase(k, key):
                        continue
                    if label:
                        following.append((label + "/" + k, node[k]))
                    else:
                        following.append((k, node[k]))
            elif isinstance(node, list):
                if key == "*":
                    for i in range(len(node)):
                        if label:
                            following.append((label + "/" + str(i), node[i]))
                        else:
                            following.append((str(i), node[i]))
                else:
                    if int(key) >= len(node):
                        sys.exit(f"query.py: index {key} out of range in {spec}")
                    following.append((label, node[int(key)]))
            else:
                sys.exit(f"query.py: {spec} goes past a value at {key!r}")
        current = following
    kept = []
    for label, value in current:
        if value is not None:
            kept.append((label, value))
    return kept


def number(v, spec):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        sys.exit(f"query.py: {spec} is not a number: {v!r}")
    return v


def numbers(specs, skip_keys):
    """The values the SPECs name, each checked to be a number."""
    values = []
    for spec in specs:
        for label, v in resolve(spec, skip_keys):
            values.append(number(v, spec))
    return values


def one_value(spec, skip_keys):
    found = resolve(spec, skip_keys)
    if len(found) != 1:
        sys.exit(f"query.py: {spec} names {len(found)} values; exactly one is needed")
    return number(found[0][1], spec)


def statistic(which, values):
    if not values:
        sys.exit("query.py: no values to take a statistic of")
    if which == "min":
        return min(values)
    if which == "max":
        return max(values)
    if which == "median":
        return statistics.median(values)
    if which == "mean":
        return statistics.mean(values)
    if which == "sum":
        return sum(values)
    if which == "count":
        return len(values)
    if which == "max-abs":
        largest = 0
        for v in values:
            if abs(v) > largest:
                largest = abs(v)
        return largest
    sys.exit(f"query.py: unknown statistic {which} (one of {', '.join(STATS)})")


def field(record, path):
    """The value at a '::' path inside a record, or None if any key is missing."""
    node = record
    for key in path.split("::"):
        if isinstance(node, dict) and key in node:
            node = node[key]
        elif isinstance(node, list) and key.isdigit() and int(key) < len(node):
            node = node[int(key)]
        else:
            return None
    return node


def records(spec, a):
    """The records a RECORDS argument names (a list counts as its items) that pass every record filter."""
    kept = []
    for label, node in resolve(spec, a.skip_key):
        items = [node]
        if isinstance(node, list):
            items = node
        for record in items:
            # record filters: all must hold
            keep = True
            for cond in a.where:
                path, val = cond.split("=", 1)
                if str(field(record, path)) != val:
                    keep = False
            for cond in a.where_in:
                path, vals = cond.split("=", 1)
                if str(field(record, path)) not in vals.split(","):
                    keep = False
            for cond in a.where_not:
                path, val = cond.split("=", 1)
                if str(field(record, path)) == val:
                    keep = False
            for path in a.nonempty:
                v = field(record, path)
                if v is None or v == "":
                    keep = False
            for cond in a.above:
                path, x = cond.split("=", 1)
                v = field(record, path)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > float(x):
                    keep = False
            for cond in a.below:
                path, x = cond.split("=", 1)
                v = field(record, path)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not v < float(x):
                    keep = False
            if keep:
                kept.append(record)
    return kept


def fmt(value, a):
    """A number at the requested precision; in tables, non-numbers print as JSON and missing values as '--'."""
    if value is None:
        return "--"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return json.dumps(value)
    if a.in_millions:
        value = value / 1e6
    if a.percent:
        value = value * 100
    if a.permille:
        value = value * 1000
    if a.sci is not None:
        return f"{value:.{a.sci}e}"
    if a.significant is not None:
        return f"{value:.{a.significant}g}"
    if a.digits is not None:
        mode = ROUND_HALF_UP
        if a.rounding == "half-even":
            mode = ROUND_HALF_EVEN
        elif a.rounding == "ceiling":
            mode = ROUND_CEILING
        d = Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-a.digits), rounding=mode)
        if a.digits < 0:
            d = d.quantize(Decimal(1))
        # no negative zero ("-0.00") for a value that rounds to zero
        if d == 0:
            d = d.copy_abs()
        if a.sep:
            return f"{d:,}"
        return str(d)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int) and a.sep:
        return f"{value:,}"
    return str(value)


def print_table(header, rows):
    """First column left-aligned, the others right-aligned, a rule under the header."""
    widths = []
    for i in range(len(header)):
        w = len(header[i])
        for row in rows:
            if len(row[i]) > w:
                w = len(row[i])
        widths.append(w)
    lines = [header]
    for row in rows:
        lines.append(row)
    for n in range(len(lines)):
        cells = []
        for i in range(len(lines[n])):
            if i == 0:
                cells.append(lines[n][i].ljust(widths[i]))
            else:
                cells.append(lines[n][i].rjust(widths[i]))
        print("  ".join(cells).rstrip())
        if n == 0:
            print("-" * (sum(widths) + 2 * (len(widths) - 1)))


# parse arguments: shared option groups (format and --skip-key, record filters, group reductions),
# then one subparser per operation
common = argparse.ArgumentParser(add_help=False)
common.add_argument("--digits", type=int, metavar="N")
common.add_argument("--rounding", choices=["half-up", "half-even", "ceiling"], default="half-up")
common.add_argument("--percent", action="store_true")
common.add_argument("--permille", action="store_true")
common.add_argument("--in-millions", action="store_true")
common.add_argument("--sep", action="store_true")
common.add_argument("--significant", type=int, metavar="N")
common.add_argument("--sci", type=int, metavar="N")
common.add_argument("--skip-key", action="append", default=[], metavar="KEY")
filters = argparse.ArgumentParser(add_help=False)
filters.add_argument("--where", action="append", default=[], metavar="FIELD=VAL")
filters.add_argument("--where-in", action="append", default=[], metavar="FIELD=V1,V2,...")
filters.add_argument("--where-not", action="append", default=[], metavar="FIELD=VAL")
filters.add_argument("--nonempty", action="append", default=[], metavar="FIELD")
filters.add_argument("--above", action="append", default=[], metavar="FIELD=X")
filters.add_argument("--below", action="append", default=[], metavar="FIELD=X")
groups = argparse.ArgumentParser(add_help=False)
groups.add_argument("--by", required=True, metavar="FIELD")
groups.add_argument("--over-groups", choices=STATS, metavar="STAT")
groups.add_argument("--exclude-largest", action="store_true")
groups.add_argument("--count-groups-above", type=float, metavar="X")
groups.add_argument("--count-groups-at-max", action="store_true")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="op", metavar="OPERATION")
sub.required = True
p = sub.add_parser("value", parents=[common])
p.add_argument("spec")
for name in ["stat", "above-one"]:
    p = sub.add_parser(name, parents=[common])
    p.add_argument("stat", choices=STATS)
    p.add_argument("specs", nargs="+")
for name in ["nth-largest", "largest-share"]:
    p = sub.add_parser(name, parents=[common])
    p.add_argument("n", type=int)
    p.add_argument("specs", nargs="+")
p = sub.add_parser("weighted-mean", parents=[common])
p.add_argument("values")
p.add_argument("weights")
p = sub.add_parser("one-over-count", parents=[common])
p.add_argument("spec")
p = sub.add_parser("log", parents=[common])
p.add_argument("base", choices=["2", "10", "e"])
p.add_argument("spec")
for name in ["diff", "ratio", "count-equal"]:
    p = sub.add_parser(name, parents=[common])
    p.add_argument("a")
    p.add_argument("b")
for name in ["diff-stat", "ratio-stat"]:
    p = sub.add_parser(name, parents=[common])
    p.add_argument("stat", choices=STATS)
    p.add_argument("specs", nargs="+")
    p.add_argument("--best", action="store_true")
p = sub.add_parser("count", parents=[filters, common])
p.add_argument("specs", nargs="+")
p.add_argument("--distinct", metavar="FIELD")
p = sub.add_parser("share", parents=[filters, common])
p.add_argument("spec")
p.add_argument("--share-where", metavar="COL=VAL")
p.add_argument("--share-where-in", metavar="COL=V1,V2,...")
p.add_argument("--share-where-not", metavar="COL=VAL")
p.add_argument("--by", metavar="FIELD")
p.add_argument("--over-groups", choices=STATS, metavar="STAT")
p = sub.add_parser("field-stat", parents=[filters, common])
p.add_argument("stat", choices=STATS)
p.add_argument("specs", nargs="+")
p.add_argument("--field", required=True, metavar="FIELD")
p = sub.add_parser("group-size", parents=[filters, groups, common])
p.add_argument("specs", nargs="+")
p.add_argument("--distinct", metavar="FIELD")
p.add_argument("--largest-share", type=int, metavar="N")
p.add_argument("--largest-group-where", metavar="COL=VAL")
p = sub.add_parser("group-stat", parents=[filters, groups, common])
p.add_argument("stat", choices=STATS)
p.add_argument("spec")
p.add_argument("--field", required=True, metavar="FIELD")
p = sub.add_parser("table", parents=[common])
p.add_argument("spec")
p.add_argument("--field", action="append", default=[], metavar="FIELD")
p = sub.add_parser("grid-table", parents=[common])
p.add_argument("grid")
p.add_argument("specs", nargs="+")
p = sub.add_parser("value-at-grid-point", parents=[common])
p.add_argument("spec")
p.add_argument("--grid", required=True)
p.add_argument("--at", required=True, type=float)
a = ap.parse_args()

# compute: each operation sets result (one number) or prints a table and exits
result = None
if a.op == "value":
    result = one_value(a.spec, a.skip_key)

elif a.op == "stat":
    values = []
    for spec in a.specs:
        for label, v in resolve(spec, a.skip_key):
            if a.stat == "count":
                values.append(v)
            else:
                values.append(number(v, spec))
    result = statistic(a.stat, values)

elif a.op == "above-one":
    values = []
    for v in numbers(a.specs, a.skip_key):
        values.append(v - 1)
    result = statistic(a.stat, values)

elif a.op == "nth-largest" or a.op == "largest-share":
    values = numbers(a.specs, a.skip_key)
    values.sort(reverse=True)
    if a.n > len(values):
        sys.exit(f"query.py: only {len(values)} values")
    if a.op == "nth-largest":
        result = values[a.n - 1]
    else:
        top = 0
        for v in values[:a.n]:
            top += v
        result = top / sum(values)

elif a.op == "weighted-mean":
    values = numbers([a.values], a.skip_key)
    weights = numbers([a.weights], a.skip_key)
    if len(values) != len(weights):
        sys.exit(f"query.py: {len(values)} values and {len(weights)} weights")
    total = 0
    weight_sum = 0
    for i in range(len(values)):
        total += values[i] * weights[i]
        weight_sum += weights[i]
    result = total / weight_sum

elif a.op == "one-over-count":
    result = 1 / len(resolve(a.spec, a.skip_key))

elif a.op == "log":
    value = one_value(a.spec, a.skip_key)
    if a.base == "2":
        result = math.log2(value)
    elif a.base == "10":
        result = math.log10(value)
    else:
        result = math.log(value)

elif a.op == "diff" or a.op == "ratio":
    first = one_value(a.a, a.skip_key)
    second = one_value(a.b, a.skip_key)
    if a.op == "diff":
        result = first - second
    else:
        result = first / second

elif a.op == "diff-stat" or a.op == "ratio-stat":
    if len(a.specs) % 2 != 0:
        sys.exit(f"query.py: {a.op} takes SPECs in pairs")
    values = []
    for i in range(0, len(a.specs), 2):
        first = numbers([a.specs[i]], a.skip_key)
        second = numbers([a.specs[i + 1]], a.skip_key)
        if a.best:
            first = [max(first)]
            second = [max(second)]
        # one side may name a single value that pairs with every value of the other side
        if len(first) == 1 and len(second) > 1:
            first = first * len(second)
        if len(second) == 1 and len(first) > 1:
            second = second * len(first)
        if len(first) != len(second) or not first:
            sys.exit(f"query.py: {a.specs[i]} and {a.specs[i + 1]} name {len(first)} and {len(second)} values")
        for j in range(len(first)):
            if a.op == "diff-stat":
                values.append(first[j] - second[j])
            else:
                values.append(first[j] / second[j])
    result = statistic(a.stat, values)

elif a.op == "count-equal":
    first = resolve(a.a, a.skip_key)
    second = resolve(a.b, a.skip_key)
    if len(first) != len(second):
        sys.exit(f"query.py: {a.a} names {len(first)} values, {a.b} names {len(second)}")
    result = 0
    for i in range(len(first)):
        if first[i][1] == second[i][1]:
            result += 1

elif a.op == "count":
    kept = []
    for spec in a.specs:
        for record in records(spec, a):
            kept.append(record)
    if a.distinct:
        seen = set()
        for record in kept:
            seen.add(str(field(record, a.distinct)))
        result = len(seen)
    else:
        result = len(kept)

elif a.op == "field-stat":
    values = []
    for spec in a.specs:
        for record in records(spec, a):
            v = field(record, a.field)
            if v is not None:
                values.append(number(v, a.field))
    result = statistic(a.stat, values)

elif a.op == "share" or a.op == "group-size" or a.op == "group-stat":
    # one value per group (share without --by: a single group)
    group_names = []
    group_values = []
    largest_group = None
    if a.op == "share":
        given = []
        if a.share_where is not None:
            given.append(a.share_where)
        if a.share_where_in is not None:
            given.append(a.share_where_in)
        if a.share_where_not is not None:
            given.append(a.share_where_not)
        if len(given) != 1:
            sys.exit("query.py: share needs exactly one of --share-where, --share-where-in, --share-where-not")
        share_col, share_val = given[0].split("=", 1)
        n_of = {}
        matching_of = {}
        for record in records(a.spec, a):
            group = ""
            if a.by is not None:
                group = str(field(record, a.by))
            if group not in n_of:
                n_of[group] = 0
                matching_of[group] = 0
            n_of[group] += 1
            v = str(field(record, share_col))
            if a.share_where is not None and v == share_val:
                matching_of[group] += 1
            if a.share_where_in is not None and v in share_val.split(","):
                matching_of[group] += 1
            if a.share_where_not is not None and v != share_val:
                matching_of[group] += 1
        if not n_of:
            sys.exit("query.py: share: no records pass the filters")
        for group in n_of:
            group_names.append(group)
            group_values.append(matching_of[group] / n_of[group])
        if a.by is None:
            result = group_values[0]
        header = [str(a.by), "share"]
    elif a.op == "group-size":
        n_records = 0
        for spec in a.specs:
            members = {}
            for record in records(spec, a):
                group = str(field(record, a.by))
                if group not in members:
                    members[group] = []
                members[group].append(record)
                n_records += 1
            for group in members:
                group_names.append(group)
                if a.distinct:
                    seen = set()
                    for record in members[group]:
                        seen.add(str(field(record, a.distinct)))
                    group_values.append(len(seen))
                else:
                    group_values.append(len(members[group]))
                # the first group in file order among those of maximal size
                if largest_group is None or len(members[group]) > len(largest_group):
                    largest_group = members[group]
        if not group_values:
            sys.exit("query.py: group-size: no records pass the filters")
        header = [a.by, "size"]
        if a.distinct:
            header = [a.by, "distinct " + a.distinct]
    else:
        members = {}
        for record in records(a.spec, a):
            v = field(record, a.field)
            if v is None:
                continue
            group = str(field(record, a.by))
            if group not in members:
                members[group] = []
            members[group].append(number(v, a.field))
        if not members:
            sys.exit("query.py: group-stat: no records pass the filters")
        for group in members:
            group_names.append(group)
            group_values.append(statistic(a.stat, members[group]))
        header = [a.by, a.stat + " " + a.field]

    # reduce the per-group values, or print them as a table
    if result is not None:
        pass
    elif a.op == "group-size" and a.largest_share is not None:
        sizes = list(group_values)
        sizes.sort(reverse=True)
        top = 0
        for s in sizes[:a.largest_share]:
            top += s
        result = top / n_records
    elif a.op == "group-size" and a.largest_group_where is not None:
        col, val = a.largest_group_where.split("=", 1)
        result = 0
        for record in largest_group:
            if str(field(record, col)) == val:
                result += 1
    else:
        if a.op != "share" and a.exclude_largest:
            largest = max(group_values)
            kept_names = []
            kept_values = []
            for i in range(len(group_values)):
                if group_values[i] != largest:
                    kept_names.append(group_names[i])
                    kept_values.append(group_values[i])
            group_names = kept_names
            group_values = kept_values
        if a.op != "share" and a.count_groups_above is not None:
            result = 0
            for v in group_values:
                if v > a.count_groups_above:
                    result += 1
        elif a.op != "share" and a.count_groups_at_max:
            largest = max(group_values)
            result = 0
            for v in group_values:
                if v == largest:
                    result += 1
        elif a.over_groups is not None:
            result = statistic(a.over_groups, group_values)
        else:
            rows = []
            for i in range(len(group_names)):
                rows.append([group_names[i], fmt(group_values[i], a)])
            print_table(header, rows)
            sys.exit(0)

if a.op == "table":
    found = resolve(a.spec, a.skip_key)
    # expand '*' in a field into one column per key of the first record that has it
    columns = []
    for path in a.field:
        if "*" not in path.split("::"):
            columns.append(path)
            continue
        head, tail = path.split("*", 1)
        head = head.rstrip(":")
        tail = tail.lstrip(":")
        for label, record in found:
            node = record
            if head:
                node = field(record, head)
            if isinstance(node, dict):
                for k in node:
                    column = str(k)
                    if head:
                        column = head + "::" + column
                    if tail:
                        column = column + "::" + tail
                    columns.append(column)
                break
    header = ["key"]
    if not columns:
        header.append("value")
    for path in columns:
        header.append(path)
    rows = []
    for label, record in found:
        row = [label]
        if not columns:
            row.append(fmt(record, a))
        for path in columns:
            row.append(fmt(field(record, path), a))
        rows.append(row)
    print_table(header, rows)
    sys.exit(0)

if a.op == "grid-table":
    found = resolve(a.grid, a.skip_key)
    if len(found) != 1:
        sys.exit(f"query.py: {a.grid} must name one list")
    grid = found[0][1]
    header = [a.grid.split("::")[-1]]
    columns = []
    for spec in a.specs:
        for label, values in resolve(spec, a.skip_key):
            if len(values) != len(grid):
                sys.exit(f"query.py: {spec} ({label}) has {len(values)} items, the grid {len(grid)}")
            if label:
                header.append(label)
            else:
                header.append(spec.split("::")[-1])
            columns.append(values)
    rows = []
    for i in range(len(grid)):
        row = [str(grid[i])]
        for values in columns:
            row.append(fmt(values[i], a))
        rows.append(row)
    print_table(header, rows)
    sys.exit(0)

if a.op == "value-at-grid-point":
    grid = resolve(a.grid, a.skip_key)[0][1]
    values = resolve(a.spec, a.skip_key)[0][1]
    if a.at not in grid:
        sys.exit(f"query.py: {a.at} is not on the grid {a.grid}")
    result = number(values[grid.index(a.at)], a.spec)

print(fmt(result, a))
