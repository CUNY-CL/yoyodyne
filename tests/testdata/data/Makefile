SUBDIRS = copy ice_g2p identity reverse tur_inflection upper

all:
	for dir in $(SUBDIRS); do $(MAKE) -C $$dir; done

clean:
	for dir in $(SUBDIRS); do $(MAKE) -C $$dir clean; done

.PHONY: all clean
