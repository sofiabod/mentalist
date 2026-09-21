class EventStream:
    def __init__(self):
        self.events = []
        self.sinks = []

    def subscribe(self, sink):
        self.sinks.append(sink)

    def emit(self, event):
        self.events.append(event)
        for sink in self.sinks:
            sink(event)
