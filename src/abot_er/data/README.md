# Data loading

Training consumes the frozen ModelScope releases. Download and verify them using
the [artifact guide](../../../data/artifacts/README.md), then prepare their local
media references using the [training guide](../../../docs/reproduction.md).

`abot_er.public_data` verifies the publication manifest; `abot_er.reproduction`
prepares shards and validates rank capacity. Internal source-data construction
pipelines are not needed to consume these releases.
