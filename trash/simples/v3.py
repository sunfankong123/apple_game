#!/usr/bin/env python
# coding: utf-8
import pygame
from pygame.locals import *
import os

SCR_RECT = Rect(0, 0, 160, 160)
bar_score = 0
FINISH_SCORE = 100
perepisode = 2000
terminal_key = False
keynum = 0
stock = 0
times = 0
last_time = 0
final= False
def main():

    pygame.init()
    screen = pygame.display.set_mode(SCR_RECT.size, 0, 32)
    # pygame.display.set_caption(u"Breakout 06 スコアの表示")

    # スプライトグループを作成して登録
    all = pygame.sprite.RenderUpdates()  # 描画用グループ
    bricks = pygame.sprite.Group()       # 衝突判定用グループ
    holes=pygame.sprite.Group()
    Paddle.containers = all
    Brick.containers = all, bricks
    Hole.containers = all ,holes

    Brick(4, 4)
    Hole(0, 0)

    # スコアボードを作成
    score_board = ScoreBoard()

    # パドル
    paddle = Paddle(bricks, holes, score_board)

    clock = pygame.time.Clock()

    global perepisode, terminal_key, times, last_time
    terminal_key = False

    # current_time = int(times / 100000)
    # if current_time != last_time and perepisode > 120:
    #    last_time = current_time
    #    perepisode -= 10
    while len(holes) > 0 and keynum < perepisode and final == False:
        clock.tick(60)
        screen.fill((0, 0, 0))
        all.update()
        all.draw(screen)
        #score_board.draw(screen)  # スコアボードを描画
        pygame.display.update()
        paddle.update()
    terminal_key = True

#    stepsfile = open("stepsdata_networks_termina/stepsdata3600000.txt", "a")
#    stepsfile.write(str(keynum) + "\n")
#    print "The agent have moved %s steps" % keynum


class Paddle(pygame.sprite.Sprite):
    """ボールを打つパドル"""
    def __init__(self, bricks, holes, score_board):
        # containersはmain()でセットされる
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image("paddle4.png")
        colorkey = self.image.get_at((0, 1))
        self.image.set_colorkey(colorkey, RLEACCEL)
        self.bricks = bricks  # ブロックグループへの参照
        self.holes = holes
        self.score_board = score_board
        self.rect.bottom = SCR_RECT.bottom  # パドルは画面の一番下
        self.rect.left = SCR_RECT.right/2

    def update(self):
        self.rect.clamp_ip(SCR_RECT)  # SCR_RECT内でしか移動できなくなる
        pygame.key.get_pressed()
        global keynum
        global times , final
        global bar_score, stock
        # 一回の移動量
        vx = vy = 16
        # すべてのブロックに対して
        for brick in self.bricks:
            # ブロックの左に重機があるとき
            if self.rect.left + 16 == brick.rect.left and self.rect.top == brick.rect.top:
                # key push
                for event in pygame.event.get():  # User did something
                    if event.type == KEYDOWN:
                        keynum += 1
                        times += 1
                        # 重機が上に移動
                        if event.key == K_UP:
                            self.rect.move_ip(0, -vy)
                            self.rect.move_ip(0, 0)
                            bar_score = 0
                        # 重機が下に移動
                        elif event.key == K_DOWN:
                            self.rect.move_ip(0, vy)
                            self.rect.move_ip(0, 0)
                            bar_score = 0
                        # 重機が右に移動　+　ブロックも右に移動
                        elif event.key == K_RIGHT:
                            self.rect.move_ip(vx, 0)
                            self.rect.move_ip(0, 0)
                            brick.rect.left += vx
                            bar_score = -1
                        # 重機が左に移動
                        elif event.key == K_LEFT:
                            self.rect.move_ip(-vx, 0)
                            self.rect.move_ip(0, 0)
                            bar_score = 0

            # ブロックの右に重機があるとき
            elif self.rect.left - 16 == brick.rect.left and self.rect.top == brick.rect.top:
                # key push
                for event in pygame.event.get():  # User did something
                    if event.type == KEYDOWN:
                        keynum += 1
                        times += 1
                        # 重機が上に移動
                        if event.key == K_UP:
                            self.rect.move_ip(0, -vy)
                            self.rect.move_ip(0, 0)
                            bar_score = 0
                        # 重機が下に移動
                        elif event.key == K_DOWN:
                            self.rect.move_ip(0, vy)
                            self.rect.move_ip(0, 0)
                            bar_score = 0
                        # 重機が右に移動
                        elif event.key == K_RIGHT:
                            self.rect.move_ip(vx, 0)
                            self.rect.move_ip(0, 0)
                            bar_score = 0
                        # 重機が左に移動　+　ブロックも左に移動
                        elif event.key == K_LEFT:
                            self.rect.move_ip(-vx, 0)
                            self.rect.move_ip(0, 0)
                            brick.rect.left -= vx
                            bar_score = 1
            # ブロックの上に重機があるとき
            elif self.rect.top + 16 == brick.rect.top and self.rect.left == brick.rect.left:
                # key push
                for event in pygame.event.get():  # User did something
                    if event.type == KEYDOWN:
                        keynum += 1
                        times += 1
                        # 重機が上に移動
                        if event.key == K_UP:
                            self.rect.move_ip(0, -vy)
                            self.rect.move_ip(0, 0)
                            bar_score = 0
                        # 重機が下に移動　+　ブロックも下に移動
                        elif event.key == K_DOWN:
                            self.rect.move_ip(0, vy)
                            self.rect.move_ip(0, 0)
                            brick.rect.top += vy
                            bar_score = 1
                        # 重機が右に移動
                        elif event.key == K_RIGHT:
                            self.rect.move_ip(vx, 0)
                            self.rect.move_ip(0, 0)
                            bar_score = 0
                        # 重機が左に移動
                        elif event.key == K_LEFT:
                            self.rect.move_ip(-vx, 0)
                            self.rect.move_ip(0, 0)
                            bar_score = 0

            # ブロックの下に重機があるとき
            elif self.rect.top - 16 == brick.rect.top and self.rect.left == brick.rect.left:
                # key push
                for event in pygame.event.get():  # User did something
                    if event.type == KEYDOWN:
                        keynum += 1
                        times += 1
                        # 重機が上に移動　+　ブロックも上に移動
                        if event.key == K_UP:
                            self.rect.move_ip(0, -vy)
                            self.rect.move_ip(0, 0)
                            brick.rect.top -= vy
                            bar_score = 1
                        # 重機が下に移動
                        elif event.key == K_DOWN:
                            self.rect.move_ip(0, vy)
                            self.rect.move_ip(0, 0)
                            bar_score = 0
                        # 重機が右に移動
                        elif event.key == K_RIGHT:
                            self.rect.move_ip(vx, 0)
                            self.rect.move_ip(0, 0)
                            bar_score = 0
                        # 重機が左に移動
                        elif event.key == K_LEFT:
                            self.rect.move_ip(-vx, 0)
                            self.rect.move_ip(0, 0)
                            bar_score = 0
            else :
                for event in pygame.event.get():  # User did something
                    if event.type == KEYDOWN:
                        keynum += 1
                        times += 1
                        bar_score = 0
                        if event.key == K_UP:
                            self.rect.move_ip(0, -vy)
                            self.rect.move_ip(0, 0)
                        elif event.key == K_DOWN:
                            self.rect.move_ip(0, vy)
                            self.rect.move_ip(0, 0)
                        elif event.key == K_RIGHT:
                            self.rect.move_ip(vx, 0)
                            self.rect.move_ip(0, 0)
                        elif event.key == K_LEFT:
                            self.rect.move_ip(-vx, 0)
                            self.rect.move_ip(0, 0)

            if brick.rect.left < 0 or brick.rect.left > 159 or brick.rect.top < 0 or brick.rect.top > 159:
                final = True
                bar_score = -5


        bricks_collided=pygame.sprite.spritecollide(brick, self.holes, True)
        if bricks_collided:
            bar_score = 10
            print "score=",bar_score



class Brick(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image("brick1.png")
        #colorkey=self.image.get_at((0,0))
        #self.image.set_colorkey(colorkey,RLEACCEL)
        # ブロックの位置を更新
        self.rect.left = SCR_RECT.left + x * self.rect.width
        self.rect.top = SCR_RECT.top + y * self.rect.height


class Hole(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image("hole.png")
        # ブロックの位置を更新
        self.rect.left = SCR_RECT.left + x * self.rect.width
        self.rect.top = SCR_RECT.top + y * self.rect.height


class ScoreBoard():
    """スコアボード"""
    def __init__(self):
        self.sysfont = pygame.font.SysFont(None, 17)
        # self.score = 0

#    def draw(self, screen):
#        score_img = self.sysfont.render("SCORE:" + str(bar_score), True, (255, 255, 255))
#        screen.blit(score_img, (1, 1))

    def add_score(self, x):
        global bar_score
        bar_score += x


def load_image(filename, colorkey=None):
    """画像をロードして画像と矩形を返す"""
    filename = os.path.join("image", filename)
    try:
        image = pygame.image.load(filename)
    except pygame.error, message:
        print "Cannot load image:", filename
        raise SystemExit, message
    image = image.convert()
    if colorkey is not None:
        if colorkey is -1:
            colorkey = image.get_at((0,0))
        image.set_colorkey(colorkey, RLEACCEL)
    return image, image.get_rect()

#turn = 0
while True:
    #turn += 1
    #if(turn == 51):
        #print "finish"
        #break
    main()

    keynum = 0
    stock = 0
    final = False
    bar_score = 0