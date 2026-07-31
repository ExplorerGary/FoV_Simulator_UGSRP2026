# Formal trace split

The CSV `FileName` column is the authoritative content label. The sidecar TXT
files contain legacy `Name = None` entries and must not be used to select the
DanceNet3D asset root.

## CircleTurns-only traces

```text
26_7_29_12_33_39.csv
26_7_29_12_35_7.csv
26_7_29_12_37_21.csv
26_7_29_12_40_25.csv
```

## GrandPlies-only traces

```text
26_7_29_12_42_57.csv
26_7_29_12_44_27.csv
26_7_29_12_48_2.csv
26_7_29_12_49_32.csv
```

On 2026-07-31, residual CircleTurns setup rows were removed from the latter
four CSVs. Each now contains only `BiancaGolden_GrandPlies` rows and covers all
1061 GrandPlies frame IDs.
