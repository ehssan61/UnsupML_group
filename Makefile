URL_1=https://archive.ics.uci.edu/static/public/562/shill+bidding+dataset.zip
data_1=shill+bidding+dataset
URL_2=https://archive.ics.uci.edu/static/public/352/online+retail.zip
data_2=online+retail

# Download and unzip datasets.
data:
	mkdir -p data
	cd data; curl -LO $(URL_1); unzip $(data_1).zip; rm -rf $(data_1).zip;
	cd data; curl -LO $(URL_2); unzip $(data_2).zip; rm -rf $(data_2).zip;

# Clean data.
clean:
	rm -rf data