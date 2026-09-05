# ANVIKSA Snake Image Manual Review Guide

## Interactive offline reviewer

From the project root, launch:

```text
python manual_snake_image_review.py
```

The application displays one case at a time, autosaves each explicit decision atomically, resumes at the first unreviewed case, and supports filters, keyboard navigation, reviewer notes, and full-size viewing. `KEEP` means technically acceptable only; it does not establish copyright or training-reuse rights.

This review resolves technical duplicate, leakage, and metadata ambiguity only. It does **not** establish copyright or training reuse rights.

## Allowed values for `human_decision`

Use exactly one of these values:

- `KEEP_TECHNICALLY_ELIGIBLE`: Images clearly show different underlying photographs or scenes.
- `EXCLUDE_DUPLICATE`: Same photograph or an obvious resized, cropped, recompressed, watermarked, or minimally edited derivative.
- `EXCLUDE_FROZEN_TEST_LEAKAGE`: Same underlying photograph or derivative as the displayed frozen-test image.
- `KEEP_LABEL_REVIEW`: Technically unique, but species metadata remains unresolved.
- `EXCLUDE_LABEL_CONFLICT`: Conflicting species metadata makes the example unsuitable.
- `KEEP_MANUAL_REVIEW`: The case remains ambiguous and should stay outside integration.
- `UNRESOLVED`: The reviewer cannot confidently decide.

Leave `human_notes` free-form. Fill `final_recommended_status` only after review; it is intentionally blank initially.

## Review method

1. Compare the underlying scene, snake pose, background geometry, cropping, and distinctive objects.
2. Treat resized, compressed, color-adjusted, watermarked, or cropped forms of the same source photograph as duplicates.
3. For frozen-test sheets, use `EXCLUDE_FROZEN_TEST_LEAKAGE` if the new image derives from the displayed test photograph.
4. Clearly different scenes, poses, backgrounds, camera positions, or animals are not duplicates merely because they show the same species.
5. For label sheets, review the conflicting directory metadata. Do not infer species or venom status from appearance.
6. When confidence is insufficient, select `KEEP_MANUAL_REVIEW` or `UNRESOLVED`; do not clear the image automatically.

## Biological-label warning

Do not use head shape, pupil shape, color, body thickness, or other simplistic visual rules to determine venom status. Directory labels remain metadata, not independently verified biological ground truth.

## Rights status

All images retain:

- `repository_provenance = VERIFIED`
- `repository_license = MIT`
- `individual_image_rights = UNCLEAR`
- `training_reuse_rights = REVIEW_REQUIRED`

Human duplicate review does not resolve copyright.
