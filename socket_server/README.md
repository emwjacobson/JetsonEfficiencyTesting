# Socket Server

`logger.py` and `verify.py` are used on a different computer where it runs as a socket server and is fed data from the benchmark application.

`logger.py` is run using python **3.9** and is the actual server.
`verify.py` is used to verify that all of the data is correctly formatted before being put into the final `data` folder.