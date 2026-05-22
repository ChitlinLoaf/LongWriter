PY=python
PKG=tools.cadence_lab
SEED=1337

.PHONY: cadence.build cadence.cluster cadence.generate cadence.all

cadence.build:
	$(PY) -m $(PKG).build_dataset --seed $(SEED)
	$(PY) -m $(PKG).features --seed $(SEED)

cadence.cluster:
	$(PY) -m $(PKG).cluster_kmeans --seed $(SEED)
	$(PY) -m $(PKG).cluster_fuzzy --seed $(SEED)

cadence.generate:
	$(PY) -m $(PKG).generator --seed $(SEED)

cadence.all: cadence.build cadence.cluster cadence.generate
