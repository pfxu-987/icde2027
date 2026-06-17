def get_task(name):
    if name == 'game24':
        from tot.tasks.game24 import Game24Task
        return Game24Task()
    if name == 'text':
        from tot.tasks.text import TextTask
        return TextTask()
    if name == 'crosswords':
        from tot.tasks.crosswords import MiniCrosswordsTask
        return MiniCrosswordsTask()
    if name == 'bird':
        from tot.tasks.bird import BirdTask
        return BirdTask()
    if name == 'wtq':
        from tot.tasks.wtq import WTQTask
        return WTQTask()
    if name == 'spider':
        from tot.tasks.spider import SpiderTask
        return SpiderTask()
    raise NotImplementedError
