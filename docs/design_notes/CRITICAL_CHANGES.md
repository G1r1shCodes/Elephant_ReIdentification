# CRITICAL PREPROCESSING REDESIGN

## ❌ ORIGINAL FATAL FLAW

**Problem:** Script cropped based on arrow tip coordinates directly.

```python
# OLD (WRONG):
center_x = arrow_tip[0]
center_y = arrow_tip[1] + 20% downward
```

**Result:**
- Arrow on back → Crop on back → **HEAD LOST** ❌
- Arrow on head → May cut ears ⚠️
- No biological awareness of identity features

---

## ✅ NEW BIOLOGICALLY-AWARE DESIGN

**Principle:** Arrow = IDENTITY SELECTOR, NOT crop target

```python
# NEW (CORRECT):
center_x = arrow_tip[0] - 10% forward bias
center_y = arrow_tip[1] - 15% upward bias
```

**Result:**
- Arrow on back → Crop BIASED toward head → Head preserved ✅
- Arrow on head → Extra room above → Ears included ✅
- Temporal gland region captured (Makhnas) ✅

---

## KEY CHANGES

### 1. Upward Bias (CRITICAL)
```python
ARROW_UPWARD_BIAS = -0.15  # 15% UPWARD (negative = up)
```
**Why:** Elephant heads are typically ABOVE/LEVEL with arrow position

### 2. Forward Bias
```python
ARROW_FORWARD_BIAS_X = -0.10  # 10% FORWARD (toward front)
```
**Why:** Identity features (ears, temporal gland) are in front anatomy

### 3. Adjusted Crop Sizes
```python
# Arrow cases: 65% × 75% (slightly smaller with bias)
# No-arrow: 85% × 85% (larger, preserve context)
```

### 4. Quality Checks
- Warns if crop < 20% of original
- Flags unusual aspect ratios
- Enables manual review of edge cases

---

## BIOLOGICAL JUSTIFICATION

### Identity Features (Priority)
1. **HEAD PROFILE** - Must preserve
2. **EARS** - Edge tears, depigmentation, shape
3. **TEMPORAL GLAND** - Makhna identifier
4. Upper torso - Secondary
5. Flanks/legs - **NOT USEFUL**

### Why Old Approach Failed
- Flank-only crops → Biometrically useless
- Missing head → Missing primary ID features
- No adaptation to arrow position

### Why New Approach Works
- Upward bias ensures head inclusion
- Forward bias captures ears/gland
- Large crops preserve context
- Quality checks catch failures

---

## VALIDATION

After running, manually verify:
- [ ] No crops missing elephant head
- [ ] At least one ear visible in all crops
- [ ] Temporal gland region visible (Makhnas)
- [ ] No flank-only or leg-only crops

---

## USAGE

```bash
python preprocessing\preprocess.py
```

Expect: ~1500+ images, all with head/ear preservation
