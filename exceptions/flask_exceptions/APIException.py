

class APIException(Exception):
    status_code = 400
    message = ""
    payload = None

    def __init__(self, message, status_code: int = None, payload=None):
        Exception.__init__(self)
        self.message = message
        if self.status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv
