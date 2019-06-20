build:
	-rm visualsearch.mar
	model-archiver --model-name visualsearch --model-path visualsearch --handler visual_service:handle
	docker build -t try-visualsearch-docker .
run:
	-make stop
	make build
	docker run -it --rm --name try-visualsearch-docker -p 8080:8080 -p 8081:8081 try-visualsearch-docker
stop:
	docker stop try-visualsearch-docker
test:
	curl -X POST http://localhost:8080/predictions/visualsearch -T kitten.jpg