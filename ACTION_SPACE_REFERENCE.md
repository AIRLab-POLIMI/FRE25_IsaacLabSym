# Action Space Quick Reference

## Before: Ternary Hidden States
```
Action Space: MultiDiscrete([3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3])
              └─steering─┘└─throttle─┘└─step─┘└──────8 hidden states─────┘

Discrete indices:  [0-2,  0-2,  0-1,  0-2,  0-2,  0-2,  0-2,  0-2,  0-2,  0-2,  0-2]
After conversion:  [-1/0/+1, -1/0/+1, 0/1, -1/0/+1, -1/0/+1, ...]

Total combinations: 3 × 3 × 2 × 3^8 = 118,098
```

## After: Binary Hidden States ✨
```
Action Space: MultiDiscrete([3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2])
              └─steering─┘└─throttle─┘└─step─┘└──────8 hidden states─────┘

Discrete indices:  [0-2,  0-2,  0-1,  0-1,  0-1,  0-1,  0-1,  0-1,  0-1,  0-1,  0-1]
After conversion:  [-1/0/+1, -1/0/+1, 0/1, -1/+1, -1/+1, -1/+1, -1/+1, ...]

Total combinations: 3 × 3 × 2 × 2^8 = 4,608
```

## Key Change
- **Hidden states**: 3 categories → 2 categories
- **Conversion**: `x - 1` → `x * 2 - 1`
- **Values**: {-1, 0, +1} → {-1, +1}
- **Complexity**: 3^8 = 6,561 → 2^8 = 256 (25.6× simpler!)

## Conversion Formulas

### Steering & Throttle (unchanged)
```python
discrete_idx ∈ {0, 1, 2}
continuous = discrete_idx - 1
result ∈ {-1, 0, +1}
```

### Step Command (unchanged)
```python
discrete_idx ∈ {0, 1}
continuous = discrete_idx
result ∈ {0, 1}
```

### Hidden States (NEW)
```python
discrete_idx ∈ {0, 1}
continuous = discrete_idx * 2 - 1
result ∈ {-1, +1}

Examples:
  0 → 0*2-1 = -1
  1 → 1*2-1 = +1
```

## Example Action

### Agent Output (discrete indices)
```python
[1, 2, 0, 1, 0, 1, 1, 0, 0, 1, 0]
```

### After Conversion
```python
[0, +1, 0, +1, -1, +1, +1, -1, -1, +1, -1]
 │   │  │  └──────────────────┬──────────────────┘
 │   │  │                     └─ 8 hidden states: {-1, +1}
 │   │  └─ step_command: {0, 1}
 │   └─ throttle: {-1, 0, +1}
 └─ steering: {-1, 0, +1}
```

## Teleop Default Values

### Discrete indices sent to environment
```python
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
 │  │  │  └────────┬────────────┘
 │  │  │           └─ 8×0 (all hidden default to 0)
 │  │  └─ 0 (E key not pressed)
 │  └─ 1 (no throttle)
 └─ 1 (no steering)
```

### After conversion
```python
[0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1]
 │  │  │  └────────┬──────────────────────┘
 │  │  │           └─ 8×(-1) (inactive state)
 │  │  └─ 0 (not stepping)
 │  └─ 0 (no throttle)
 └─ 0 (no steering)
```

## Comparison Table

| Property | Before (Ternary) | After (Binary) | Change |
|----------|------------------|----------------|--------|
| Hidden categories | 3 | 2 | -33% |
| Hidden values | {-1, 0, +1} | {-1, +1} | No neutral |
| Hidden combinations | 6,561 | 256 | -96% |
| Total combinations | 118,098 | 4,608 | -96% |
| Conversion formula | `x - 1` | `x * 2 - 1` | Simpler |
| Teleop default | 0 (neutral) | -1 (inactive) | Changed |
