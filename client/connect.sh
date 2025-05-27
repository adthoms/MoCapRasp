sudo pkill ptpd
sudo ptpd -g -s -i eth0 --statistics-file /tmp/ptpd-stats.txt