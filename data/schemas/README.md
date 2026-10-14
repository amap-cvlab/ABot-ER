# Record schema policy

The canonical schema will be migrated from the sealed loader and release. Required concepts: identity, source and split, source revision, group/episode/scene/video IDs where applicable, media references and hashes, original question hash, conversations, task family, benchmark proxy, output-contract ID and data version.

Do not invent missing provenance or simulator validation flags. Use explicit unknown/pending states. Prefer plural `images` and `videos`; validate their placeholders. Preserve original annotation and transformation provenance separately from normalized labels.
