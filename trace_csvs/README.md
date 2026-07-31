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
26_7_31_14_59_37.csv
26_7_31_15_1_21.csv
26_7_31_15_3_19.csv
26_7_31_15_5_13.csv
26_7_31_15_6_30.csv
26_7_31_15_7_7.csv
```

These ten complete files are the formal trace-level dataset for the linear
prediction experiment. The split is performed by CSV file, never by row.

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
