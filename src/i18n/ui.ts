import type { Language } from '@/i18n/config'

interface Translation {
  title: string
  subtitle: string
  description: string
  posts: string
  tags: string
  about: string
  toc: string
}

export const ui: Record<Language, Translation> = {
  'de': {
    title: 'Blüten der Morgenröte',
    subtitle: 'Essays, Momentaufnahmen und Gedanken von zhuz1',
    description: 'shenshijiu_com ist der persönliche Blog von zhuz1. Die Website basiert auf dem Astro-Framework, verwendet das statische Blog-Theme Retypeset und wird über CloudFlare bereitgestellt.',
    posts: 'Beiträge',
    tags: 'Schlagwörter',
    about: 'Über',
    toc: 'Inhaltsverzeichnis',
  },
  'en': {
    title: 'Morning Blossoms at Dusk',
    subtitle: 'Essays, snapshots, and passing thoughts by zhuz1',
    description: 'shenshijiu_com is zhuz1’s personal blog, built with Astro, powered by the Retypeset static blog theme, and deployed on CloudFlare.',
    posts: 'Posts',
    tags: 'Tags',
    about: 'About',
    toc: 'Table of Contents',
  },
  'es': {
    title: 'Flores de la mañana al atardecer',
    subtitle: 'Ensayos, instantáneas y pensamientos sueltos de zhuz1',
    description: 'shenshijiu_com es el blog personal de zhuz1, creado con Astro, usando el tema estático Retypeset y desplegado en CloudFlare.',
    posts: 'Artículos',
    tags: 'Etiquetas',
    about: 'Sobre',
    toc: 'Índice',
  },
  'fr': {
    title: 'Fleurs du matin au crépuscule',
    subtitle: 'Essais, instantanés et pensées éparses de zhuz1',
    description: 'shenshijiu_com est le blog personnel de zhuz1, conçu avec Astro, utilisant le thème statique Retypeset et déployé sur CloudFlare.',
    posts: 'Articles',
    tags: 'Étiquettes',
    about: 'À propos',
    toc: 'Table des matières',
  },
  'ja': {
    title: '朝花夕拾',
    subtitle: 'zhuz1 の随筆、スナップ、雑記',
    description: 'shenshijiu_com は zhuz1 の個人ブログです。Astro フレームワークを基盤とし、Retypeset の静的ブログテーマを採用し、CloudFlare 上にデプロイされています。',
    posts: '記事',
    tags: 'タグ',
    about: '概要',
    toc: '目次',
  },
  'ko': {
    title: '아침꽃을 저녁에 줍다',
    subtitle: 'zhuz1의 수필, 스냅사진, 그리고 단상들',
    description: 'shenshijiu_com은 zhuz1의 개인 블로그로, Astro 프레임워크를 기반으로 Retypeset 정적 블로그 테마를 사용하며 CloudFlare에 배포되어 있습니다.',
    posts: '게시물',
    tags: '태그',
    about: '소개',
    toc: '목차',
  },
  'pl': {
    title: 'Poranne kwiaty o zmierzchu',
    subtitle: 'Eseje, migawki i luźne myśli zhuz1',
    description: 'shenshijiu_com to osobisty blog zhuz1, oparty na frameworku Astro, wykorzystujący statyczny motyw Retypeset i wdrożony na CloudFlare.',
    posts: 'Artykuły',
    tags: 'Tagi',
    about: 'O stronie',
    toc: 'Spis treści',
  },
  'pt': {
    title: 'Flores da manhã ao entardecer',
    subtitle: 'Ensaios, registros e pensamentos soltos de zhuz1',
    description: 'shenshijiu_com é o blog pessoal de zhuz1, criado com Astro, usando o tema estático Retypeset e implantado na CloudFlare.',
    posts: 'Artigos',
    tags: 'Tags',
    about: 'Sobre',
    toc: 'Sumário',
  },
  'ru': {
    title: 'Утренние цветы на закате',
    subtitle: 'Эссе, снимки и мимолётные мысли zhuz1',
    description: 'shenshijiu_com — это личный блог zhuz1, созданный на базе Astro, с использованием статической темы Retypeset и размещённый на CloudFlare.',
    posts: 'Посты',
    tags: 'Теги',
    about: 'О себе',
    toc: 'Оглавление',
  },
  'zh': {
    title: '朝花夕拾',
    subtitle: 'zhuz1 的随笔、随拍、碎碎念',
    description: 'shenshijiu_com 是 zhuz1 的个人博客网站，基于 Astro 框架，使用 Retypeset 静态博客主题，并部署在 CloudFlare 上。',
    posts: '文章',
    tags: '标签',
    about: '关于',
    toc: '目录',
  },
  'zh-tw': {
    title: '朝花夕拾',
    subtitle: 'zhuz1 的隨筆、隨拍、碎碎念',
    description: 'shenshijiu_com 是 zhuz1 的個人部落格網站，基於 Astro 框架，使用 Retypeset 靜態部落格主題，並部署於 CloudFlare 上。',
    posts: '文章',
    tags: '標籤',
    about: '關於',
    toc: '目錄',
  },
}
