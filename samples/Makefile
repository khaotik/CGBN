SUBDIRS = \
	sample_01_add \
	sample_02_modinv \
	sample_03_powm \
	sample_04_miller_rabin

all: $(SUBDIRS)

.PHONY: clean $(SUBDIRS)
$(SUBDIRS):
	make -C $@

clean:
	$(foreach subdir, $(SUBDIRS), \
	  make -C $(subdir) clean; \
	)
