#!/usr/bin/env python
# coding: utf-8
import pygame
from pygame.locals import *
import os
import random
import math
from math import *

screen_x = 128
screen_y = 128
SCR_RECT = Rect(0, 0, screen_x, screen_y)
bar_score = 0
FINISH_SCORE = 100
perepisode = 700
terminal_key = False
keynum = 0
stock = 0
times = 0
last_time = 0
final = False
angle = 360
# stepsfile = open("stepsdata/stepsdata.txt", "a")


def main():
    pygame.init()
    screen = pygame.display.set_mode(SCR_RECT.size, 0, 32)
    # pygame.display.set_caption(u"Breakout 06 スコアの表示")

    # スプライトグループを作成して登録
    all = pygame.sprite.OrderedUpdates()  # 描画用グループ
    bricks = pygame.sprite.Group()  # 衝突判定用グループ
    holes = pygame.sprite.Group()
    Brick.containers = all, bricks
    Hole.containers = all, holes
    Paddle.containers = all

    for x in range(6,26, 1 ):
        Hole(x, 0, 0)
    for x in range(6, 26, 1):
        Hole(x, 1, 0)
    for x in range(7, 25, 1):
        Hole(x, 2, 0)
    # Hole(35, 0, 0)
    # Hole(5, 0, 0)
    # Hole(6, 0, 0)
    # Hole(7, 0, 0)
    # Hole(8, 0, 0)
    # Hole(9, 0, 0)
    #
    # Hole(10, 0, 0)
    # Hole(11, 0, 0)
    # Hole(12, 0, 0)
    # Hole(3, 0, 0)
    #
    # Hole(4, 1, 0)
    # Hole(5, 1, 0)
    # Hole(6, 1, 0)
    # Hole(7, 1, 0)
    # Hole(8, 1, 0)
    # Hole(9, 1, 0)
    #
    # Hole(10, 1, 0)
    # Hole(11, 1, 0)
    # Hole(12, 1, 0)
    # Hole(3, 1, 0)

    brick_random_x = random.randint(-1, 1) * 6
    brick_random_y = random.randint(-1, 0) * 3

    brickX = 25
    brickY = 20

    Brick(brickX - 6 + brick_random_x, brickY + brick_random_y, 1)
    Brick(brickX - 6 + brick_random_x, brickY + brick_random_y + 1, 1)
    Brick(brickX - 6 + brick_random_x, brickY + brick_random_y + 2, 1)
    Brick(brickX - 6 + brick_random_x, brickY + brick_random_y + 3, 1)
    Brick(brickX - 6 + brick_random_x, brickY + brick_random_y + 4, 1)
    Brick(brickX - 6 + brick_random_x, brickY + brick_random_y + 5, 1)
    Brick(brickX - 6 + brick_random_x, brickY + brick_random_y - 1, 1)
    Brick(brickX - 7 + brick_random_x, brickY + brick_random_y - 1, 1)
    Brick(brickX - 8 + brick_random_x, brickY + brick_random_y - 1, 1)
    Brick(brickX - 9 + brick_random_x, brickY + brick_random_y - 1, 1)
    Brick(brickX - 10 + brick_random_x, brickY + brick_random_y - 1, 1)
    Brick(brickX - 11 + brick_random_x, brickY + brick_random_y - 1, 1)
    Brick(brickX - 12 + brick_random_x, brickY + brick_random_y - 1, 1)
    Brick(brickX - 12 + brick_random_x, brickY + brick_random_y, 1)
    Brick(brickX - 12 + brick_random_x, brickY + brick_random_y + 1, 1)
    Brick(brickX - 12 + brick_random_x, brickY + brick_random_y + 2, 1)
    Brick(brickX - 12 + brick_random_x, brickY + brick_random_y + 3, 1)
    Brick(brickX - 12 + brick_random_x, brickY + brick_random_y + 4, 1)
    Brick(brickX - 12 + brick_random_x, brickY + brick_random_y + 5, 1)
    Brick(brickX - 11 + brick_random_x, brickY + brick_random_y + 5, 1)
    Brick(brickX - 10 + brick_random_x, brickY + brick_random_y + 5, 1)
    Brick(brickX - 9 + brick_random_x, brickY + brick_random_y + 5, 1)
    Brick(brickX - 8 + brick_random_x, brickY + brick_random_y + 5, 1)
    Brick(brickX - 7 + brick_random_x, brickY + brick_random_y + 5, 1)

    Brick(brickX - 11 + brick_random_x, brickY + brick_random_y, 1)
    Brick(brickX - 11 + brick_random_x, brickY + brick_random_y + 1, 1)
    Brick(brickX - 11 + brick_random_x, brickY + brick_random_y + 2, 1)
    Brick(brickX - 11 + brick_random_x, brickY + brick_random_y + 3, 1)
    Brick(brickX - 11 + brick_random_x, brickY + brick_random_y + 4, 1)
    Brick(brickX - 10 + brick_random_x, brickY + brick_random_y, 1)
    Brick(brickX - 10 + brick_random_x, brickY + brick_random_y + 1, 1)
    Brick(brickX - 10 + brick_random_x, brickY + brick_random_y + 2, 1)
    Brick(brickX - 10 + brick_random_x, brickY + brick_random_y + 3, 1)
    Brick(brickX - 10 + brick_random_x, brickY + brick_random_y + 4, 1)
    Brick(brickX - 9 + brick_random_x, brickY + brick_random_y, 1)
    Brick(brickX - 9 + brick_random_x, brickY + brick_random_y + 1, 1)
    Brick(brickX - 9 + brick_random_x, brickY + brick_random_y + 2, 1)
    Brick(brickX - 9 + brick_random_x, brickY + brick_random_y + 3, 1)
    Brick(brickX - 9 + brick_random_x, brickY + brick_random_y + 4, 1)
    Brick(brickX - 8 + brick_random_x, brickY + brick_random_y, 1)
    Brick(brickX - 8 + brick_random_x, brickY + brick_random_y + 1, 1)
    Brick(brickX - 8 + brick_random_x, brickY + brick_random_y + 2, 1)
    Brick(brickX - 8 + brick_random_x, brickY + brick_random_y + 3, 1)
    Brick(brickX - 8 + brick_random_x, brickY + brick_random_y + 4, 1)
    Brick(brickX - 7 + brick_random_x, brickY + brick_random_y, 1)
    Brick(brickX - 7 + brick_random_x, brickY + brick_random_y + 1, 1)
    Brick(brickX - 7 + brick_random_x, brickY + brick_random_y + 2, 1)
    Brick(brickX - 7 + brick_random_x, brickY + brick_random_y + 3, 1)
    Brick(brickX - 7 + brick_random_x, brickY + brick_random_y + 4, 1)

    Brick(brickX - 8 + brick_random_x, brickY + brick_random_y + 1, 1)
    Brick(brickX - 8 + brick_random_x, brickY + brick_random_y + 2, 1)
    Brick(brickX - 8 + brick_random_x, brickY + brick_random_y + 3, 1)
    Brick(brickX - 9 + brick_random_x, brickY + brick_random_y + 1, 1)
    Brick(brickX - 9 + brick_random_x, brickY + brick_random_y + 2, 1)
    Brick(brickX - 9 + brick_random_x, brickY + brick_random_y + 3, 1)
    Brick(brickX - 10 + brick_random_x, brickY + brick_random_y + 1, 1)
    Brick(brickX - 10 + brick_random_x, brickY + brick_random_y + 2, 1)
    Brick(brickX - 10 + brick_random_x, brickY + brick_random_y + 3, 1)

    # パドル
    paddle = Paddle(bricks, holes, angle, all)

    clock = pygame.time.Clock()

    global perepisode, terminal_key, times, last_time, bar_score, final
    terminal_key = False

    # current_time = int(times / 100000)
    # if current_time != last_time and perepisode > 120:
    #    last_time = current_time
    #    perepisode -= 10
    while len(holes) > 0 and keynum < perepisode and final is False:
        # print "holes=", len(holes)
        # print "bricks=", len(bricks)
        clock.tick(60)
        screen.fill((51, 51, 51))
        all.update()
        all.draw(screen)
        pygame.display.update()
    terminal_key = True
#    stepsfile.write(str(keynum) + "\n")
#    print "The agent have moved %s steps" % keynum


class Paddle(pygame.sprite.Sprite):
    """ボールを打つパドル"""

    def __init__(self, bricks, holes, angle, all):
        # containersはmain()でセットされる
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image("paddle4.png")
        # colorkey = self.image.get_at((0, 2))
        # self.image.set_colorkey(colorkey, RLEACCEL)
        self._image = self.image
        self.forward = 4
        self.back = 4
        self.angle = angle
        self.bricks = bricks  # ブロックグループへの参照
        self.holes = holes
        self.alls = all
        self.rect.center = (72, 120)
        self.x, self.y = self.rect.center

    def rotate(self):
        center = self.rect.center
        self.image = pygame.transform.rotate(self._image, self.angle)
        self.rect = self.image.get_rect(center=center)

    def update(self):
        # self.rect.clamp_ip(SCR_RECT)  # SCR_RECT内でしか移動できなくなる
        turn_speed = 5
        self._rect = Rect(self.rect)
        self._rect.center = self.x, self.y
        # turn_speed = 90
        # pygame.key.get_pressed()
        global keynum
        global times
        global bar_score, stock
        num = 0

        numa = 0
        numb = 0
        numc = 0
        numd = 0
        nume = 0
        numf = 0
        numg = 0
        numh = 0

        a = []
        b = []
        c = []
        d = []
        e = []
        f = []
        g = []
        h = []
        keyup = False
        keydown = False
        keyright = False
        keyleft = False
        # すべてのブロックに対して
        # 一回の移動量
        vx = vy = 4
        for event in pygame.event.get():  # User did something
            if event.type == KEYDOWN:
                # print bar_score
                bar_score = 0
                keynum += 1
                times += 1
                # print "--------------------------"
                # 重機の操縦(初代+旋回simple)
                if event.key == K_UP:
                    keyup = True
                    self.rect.move_ip(0, -vy)
                    self.angle = 0
                elif event.key == K_DOWN:
                    keydown = True
                    self.rect.move_ip(0, vy)
                    self.angle = 180
                elif event.key == K_RIGHT:
                    keyright = True
                    self.rect.move_ip(vx, 0)
                    self.angle = 270
                elif event.key == K_LEFT:
                    keyleft = True
                    self.rect.move_ip(-vx, 0)
                    self.angle = 90

                    """
                if event.key == K_UP:
                    self.rect.left += sin(radians(self.angle)) * -self.forward
                    self.rect.top += cos(radians(self.angle)) * -self.forward
                    keyup = True
                elif event.key == K_DOWN:
                    self.rect.left += sin(radians(self.angle)) * self.back
                    self.rect.top += cos(radians(self.angle)) * self.back
                elif event.key == K_RIGHT:
                    self.angle -= turn_speed
                elif event.key == K_LEFT:
                    self.angle += turn_speed
                """

            # 重機と衝突したブロックリストを取得
            bricks_collided = pygame.sprite.spritecollide(self, self.bricks, False)
            holes_collided = pygame.sprite.spritecollide(self, self.holes, False)
            if holes_collided:
                if keyup == True:
                    self.rect.top += 4
                    # elif keydown == True:
                    # self.rect.top -= 4
                elif keyright == True:
                    self.rect.left -= 4
                elif keyleft == True:
                    self.rect.left += 4
                break
            if bricks_collided:  # 衝突ブロックがある場合
                # print "-------------------------------------------------------"
                # print "X＝", self.rect.left / 32
                # print "Y＝", self.rect.top / 32


                for brick in bricks_collided:  # 各衝突ブロックに対して
                    # print "-------------"
                    # print "X＝",brick.rect.left / 32
                    # print "Y＝",brick.rect.top / 32
                    # print "-------------"
                    if keyup == True:
                        brick.rect.top -= 4
                        bar_score += 1
                    elif keydown == True:
                        brick.rect.top += 4
                        bar_score -= 1
                    elif keyright == True:
                        brick.rect.left += 4
                    elif keyleft == True:
                        brick.rect.left -= 4

                    if brick.rect.left == self.rect.left:
                        a = pygame.sprite.spritecollide(brick, self.bricks, False)
                    elif brick.rect.left == self.rect.left + brick.rect.width:
                        b = pygame.sprite.spritecollide(brick, self.bricks, False)
                    elif brick.rect.left == self.rect.left + 2 * brick.rect.width:
                        c = pygame.sprite.spritecollide(brick, self.bricks, False)
                    elif brick.rect.left == self.rect.left + 3 * brick.rect.width:
                        d = pygame.sprite.spritecollide(brick, self.bricks, False)
                    elif brick.rect.top == self.rect.top:
                        e = pygame.sprite.spritecollide(brick, self.bricks, False)
                    elif brick.rect.top == self.rect.top + brick.rect.height:
                        f = pygame.sprite.spritecollide(brick, self.bricks, False)
                    elif brick.rect.top == self.rect.top + 2 * brick.rect.height:
                        g = pygame.sprite.spritecollide(brick, self.bricks, False)
                    elif brick.rect.top == self.rect.top + 3 * brick.rect.height:
                        h = pygame.sprite.spritecollide(brick, self.bricks, False)

                    # print '  ', len(a), len(b), len(c), len(d), '  '
                    # print len(e), '  ', '  ', '  ', '  ', len(e)
                    # print len(f), '  ', '  ', '  ', '  ', len(f)
                    # print len(g), '  ', '  ', '  ', '  ', len(g)
                    # print len(h), '  ', '  ', '  ', '  ', len(h)
                    # print '  ', len(a), len(b), len(c), len(d), '  '

                    # 四辺の判定
                    if brick.rect.left < 0:
                        brick.rect.left = 0
                        # bar_score -= 20
                        self.rect.left = 4
                        # bar_score -= 5
                    if brick.rect.top < 0:
                        brick.rect.top = 0
                        self.rect.top = 4
                        # bar_score -= 1
                    if brick.rect.right > screen_x:
                        brick.rect.right = screen_x
                        self.rect.left = screen_x - self.rect.width - 4
                        # bar_score -= 5
                        # bar_score -= 20
                    if brick.rect.bottom > screen_y:
                        brick.rect.bottom = screen_y
                        self.rect.top = screen_y - self.rect.height - 4
                        # bar_score -= 20
                        # bar_score -= 5

                    # brickとholeの衝突判定
                    bricks_collided2 = pygame.sprite.spritecollide(brick, self.holes, False)
                    if bricks_collided2:
                        pygame.sprite.Group.remove(self.bricks, brick)
                        pygame.sprite.Group.remove(self.alls, brick)
                        # 衝突しているbrickとholeをgroupから削除
                        for hole in bricks_collided2:
                            pygame.sprite.Group.remove(self.holes, hole)
                            pygame.sprite.Group.remove(self.alls, hole)
                            bar_score += 20
                            break

                # 土砂シミュレーション
                # Center
                if len(a) > 4:
                    for brick in bricks_collided:
                        if self.rect.left == brick.rect.left or numa % 2 == 1:
                            numa += 1
                            brick.rect.left -= 4
                            if len(a) - numa == 4:
                                break
                        elif self.rect.left == brick.rect.left:
                            numa += 1
                            brick.rect.left += 4
                            if len(a) - numa == 4:
                                break

                if len(b) > 4:
                    for brick in bricks_collided:
                        if self.rect.left == brick.rect.left - brick.rect.width:
                            if numb % 2 == 1:
                                numb += 1
                                brick.rect.left -= 4
                                # print numb
                                if len(b) - numb == 4:
                                    break
                            else:
                                numb += 1
                                brick.rect.left += 4
                                # print "----------------------"
                                if len(b) - numb == 4:
                                    break

                if len(c) > 4:
                    for brick in bricks_collided:
                        # print self.rect.left, brick.rect.left
                        if self.rect.left == brick.rect.left - 2 * brick.rect.width:
                            if numc % 2 == 1:
                                numc += 1
                                brick.rect.left += 4
                                if len(c) - numc == 4:
                                    break
                            else:
                                # print "---"
                                numc += 1
                                brick.rect.left -= 4
                                if len(c) - numc == 4:
                                    break

                if len(d) > 4:
                    for brick in bricks_collided:
                        if self.rect.left == brick.rect.left - 3 * brick.rect.width:
                            if numd % 2 == 1:
                                numd += 1
                                brick.rect.left += 4
                                if len(d) - numd == 4:
                                    break
                            else:
                                numd += 1
                                brick.rect.left -= 4
                                if len(d) - numd == 4:
                                    break
                # Side
                if len(e) > 4:
                    for brick in bricks_collided:
                        if self.rect.top == brick.rect.top:
                            if nume % 2 == 1:
                                nume += 1
                                brick.rect.top -= 4
                                if len(e) - nume == 4:
                                    break
                            else:
                                nume += 1
                                brick.rect.top += 4
                                if len(e) - nume == 4:
                                    break

                if len(f) > 4:
                    for brick in bricks_collided:
                        if self.rect.top == brick.rect.top - brick.rect.height:
                            if numf % 2 == 1:
                                numf += 1
                                brick.rect.top -= 4
                                if len(f) - numf == 4:
                                    break
                            else:
                                numf += 1
                                brick.rect.top += 4
                                if len(f) - numf == 4:
                                    break

                if len(g) > 4:
                    for brick in bricks_collided:
                        if self.rect.top == brick.rect.top - 2 * brick.rect.height:
                            if numg % 2 == 1:
                                numg += 1
                                brick.rect.top += 4
                                if len(g) - numg == 4:
                                    break
                            else:
                                numg += 1
                                brick.rect.top -= 4
                                if len(g) - numg == 4:
                                    break

                if len(h) > 4:
                    for brick in bricks_collided:
                        if self.rect.top == brick.rect.top - 3 * brick.rect.height:
                            if numh % 2 == 1:
                                numh += 1
                                brick.rect.top += 4
                                if len(h) - numh == 4:
                                    break
                            else:
                                numh += 1
                                brick.rect.top -= 4
                                if len(h) - numh == 4:
                                    break
                for brick in bricks_collided:
                    # 四辺の判定
                    if brick.rect.left < 0:
                        brick.rect.left = 0
                    if brick.rect.top < 0:
                        brick.rect.top = 0
                    if brick.rect.right > screen_x:
                        brick.rect.right = screen_x
                    if brick.rect.bottom > screen_y:
                        brick.rect.bottom = screen_y

                # brickとholeの衝突判定
                bricks_collided2 = pygame.sprite.spritecollide(brick, self.holes, False)
                if bricks_collided2:
                    pygame.sprite.Group.remove(self.bricks, brick)
                    pygame.sprite.Group.remove(self.alls, brick)
                    # 衝突しているbrickとholeをgroupから削除
                    for hole in bricks_collided2:
                        pygame.sprite.Group.remove(self.holes, hole)
                        pygame.sprite.Group.remove(self.alls, hole)
                        bar_score += 15
                        break
                """
            # brick同士の衝突(取り残される処理)
            elif len(bricks_collided) > 3:
                for brick in bricks_collided:
                    num += 1
                    if keyup == True:
                        brick.rect.top += 8
                    if keydown == True:
                        brick.rect.top -= 8
                    if keyright == True:
                        brick.rect.left -= 8
                    if keyleft == True:
                        brick.rect.left += 8
                    if len(bricks_collided) - num == 3:
                        break
            """
            # 土砂の高さをdepthに変換
            for brick in self.bricks:
                old_x = brick.rect.left / 4
                old_y = brick.rect.top / 4
                pygame.sprite.Group.remove(self.bricks, brick)
                pygame.sprite.Group.remove(self.alls, brick)
                a = pygame.sprite.spritecollide(brick, self.bricks, False)
                Brick(old_x, old_y, len(a))

            # 穴の深さをdepthに変換
            for hole in self.holes:
                old_x = hole.rect.left / 4
                old_y = hole.rect.top / 4
                pygame.sprite.Group.remove(self.holes, hole)
                pygame.sprite.Group.remove(self.alls, hole)
                b = pygame.sprite.spritecollide(hole, self.holes, False)
                Hole(old_x, old_y, len(b))

            self.rect.clamp_ip(SCR_RECT)
            pygame.sprite.Group.remove(self.alls, self)
            pygame.sprite.Group.add(self.alls, self)
            # 穴を全部埋めたら報酬大
            if len(self.holes) == 0:
                bar_score += 100
            # print "score=", bar_score
            # print "hole=",len(self.holes)
            self.rotate()


class Brick(pygame.sprite.Sprite):
    def __init__(self, x, y, level):
        pygame.sprite.Sprite.__init__(self, self.containers)
        global bar_score, terminal_key
        self.level = level
        self.depth = 51 + (self.level + 1) * 51
        if self.depth > 255:
            self.depth = 255
        self.image = pygame.Surface((4, 4), 0, 32)
        self.rect = self.image.fill((self.depth, self.depth, self.depth))
        # ブロックの位置を更新
        self.rect.left = SCR_RECT.left + x * self.rect.width
        self.rect.top = SCR_RECT.top + y * self.rect.height


class Hole(pygame.sprite.Sprite):
    def __init__(self, x, y, level):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.level = level
        self.depth = 51 - (self.level + 1) * 51
        self.image = pygame.Surface((4, 4), 0, 32)
        self.rect = self.image.fill((self.depth, self.depth, self.depth))
        # ブロックの位置を更新
        self.rect.left = SCR_RECT.left + x * self.rect.width
        self.rect.top = SCR_RECT.top + y * self.rect.height


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
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)
    return image, image.get_rect()


turn = 0
# s_turn = 0
while True:
    #    s_turn += 1
    #    if s_turn == 10:
    #        print "finish"
    #        stepsfile.write("finish\n")
    #        s_turn = 0
    #        turn += 1
    main()

    keynum = 0
    stock = 0
    final = False
